import os
import yaml
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 导入框架组件
from openpan.registry import MODELS
import openpan.models.backbones.dynamic_conv
import openpan.models.necks.sft_neck
import openpan.models.heads.dynamic_head
import openpan.models.framework

def linear_stretch(image, percent=2.0):
    """专业的遥感图像 2%~98% 线性拉伸，去除极值，增强对比度"""
    stretched = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[-1]):
        band = image[:, :, i]
        valid_mask = band > 0
        if valid_mask.sum() > 0:
            valid_pixels = band[valid_mask]
            low = np.percentile(valid_pixels, percent)
            high = np.percentile(valid_pixels, 100 - percent)
            if high > low:
                stretched[:, :, i] = np.clip((band - low) / (high - low), 0, 1)
            else:
                stretched[:, :, i] = band
    return stretched

def to_false_color(tensor_img, is_pan=False):
    """将 Tensor 转换为假彩色 numpy 图像 (NIR-Red-Green)"""
    img_np = tensor_img.squeeze(0).cpu().numpy()
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    else:
        img_np = np.expand_dims(img_np, axis=-1)

    if is_pan or img_np.shape[-1] == 1:
        stretched = linear_stretch(img_np)
        return np.repeat(stretched, 3, axis=-1)

    c = img_np.shape[-1]
    if c >= 4:
        bands = [3, 2, 1] # 提取 NIR(3), R(2), G(1)
        false_color_img = img_np[:, :, bands]
    elif c == 3:
        false_color_img = img_np
    else:
        false_color_img = img_np[:, :, :3]

    return linear_stretch(false_color_img)

def calculate_psnr(img1, img2):
    """计算峰值信噪比 (PSNR)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()

def main():
    # ================== 配置区域 ==================
    config_path = "configs/models/unsupervised_dynamic_pan.yaml"
    checkpoint_path = "work_dirs/unsupervised_dynamic_pan/latest.pth" # 或者 best.pth
    test_h5_path = "D:/Nanqing/DeepLearning/Landsat_Pansharpen/datasets_h5/pansharpening_gf2/test_data/test_gf2_multiExm.h5"
    test_index = 10 # 随意挑选 H5 文件中的第几张图像进行测试 (比如第 0, 10, 50 张)
    max_value = 2047.0 
    # ==============================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 1. 初始化模型并加载权重
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model = MODELS.build(cfg['model']).to(device)

    if os.path.exists(checkpoint_path):
        print(f"📦 Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}. Using untrained model!")

    # 2. 读取数据 (自动寻找 GT)
    has_gt = False
    with h5py.File(test_h5_path, 'r') as f:
        ms_np = f['ms'][test_index].astype(np.float32) / max_value
        pan_np = f['pan'][test_index].astype(np.float32) / max_value
        
        # 尝试读取真值 (Ground Truth)
        if 'gt' in f:
            gt_np = f['gt'][test_index].astype(np.float32) / max_value
            has_gt = True
            
    ms_tensor = torch.from_numpy(ms_np).unsqueeze(0).to(device)
    pan_tensor = torch.from_numpy(pan_np).unsqueeze(0).to(device)
    if has_gt:
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).to(device)

    # 3. 模型推理
    print("🧠 Running Inference...")
    with torch.no_grad():
        ms_up = F.interpolate(ms_tensor, size=pan_tensor.shape[-2:], mode='bicubic', align_corners=False)
        pred_hrms = model(ms_tensor, pan_tensor)
        
        pred_hrms = torch.clamp(pred_hrms, 0, 1)
        ms_up = torch.clamp(ms_up, 0, 1)

    # 4. 定量指标评估 (拿 GT 做裁判)
    if has_gt:
        psnr_val = calculate_psnr(pred_hrms, gt_tensor)
        print(f"\n📊 --- Quantitative Evaluation ---")
        print(f"   Model vs Ground Truth PSNR : {psnr_val:.2f} dB")
        print(f"   (Bicubic vs Ground Truth PSNR: {calculate_psnr(ms_up, gt_tensor):.2f} dB)")
        print(f"----------------------------------\n")

    # 5. 可视化生成
    print("🎨 Generating False Color Visualizations...")
    vis_ms = to_false_color(ms_tensor)
    vis_pan = to_false_color(pan_tensor, is_pan=True)
    vis_pred = to_false_color(pred_hrms)
    if has_gt:
        vis_gt = to_false_color(gt_tensor)

    # 6. 使用 Matplotlib 绘图 (根据是否有 GT 动态调整列数)
    num_cols = 4 if has_gt else 3
    plt.figure(figsize=(5 * num_cols, 5))
    plt.suptitle("Pansharpening Evaluation (False Color: NIR-Red-Green)", fontsize=16, fontweight='bold')

    plt.subplot(1, num_cols, 1)
    plt.title("LRMS (Bicubic Upsampled)")
    plt.imshow(vis_ms)
    plt.axis('off')

    plt.subplot(1, num_cols, 2)
    plt.title("PAN (High Resolution)")
    plt.imshow(vis_pan, cmap='gray')
    plt.axis('off')

    plt.subplot(1, num_cols, 3)
    plt.title("Fused HRMS (Model Output)")
    plt.imshow(vis_pred)
    plt.axis('off')

    if has_gt:
        plt.subplot(1, num_cols, 4)
        plt.title("Ground Truth (Reference)")
        plt.imshow(vis_gt)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()