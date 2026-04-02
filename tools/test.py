import os
import yaml
import torch
from tqdm import tqdm

# ==========================================
# 🌟 强制导入以激活 Registry
# ==========================================
import openpan.datasets.h5_dataset
import openpan.models.backbones.dynamic_conv
import openpan.models.necks.sft_neck
import openpan.models.heads.dynamic_head
import openpan.models.losses.unsupervised_loss
import openpan.models.framework

from openpan.registry import MODELS, DATASETS
from torch.utils.data import DataLoader

# 🌟 从你刚刚新建的解耦模块中导入所有指标函数
from openpan.evaluation.metrics import (
    calculate_psnr, calculate_ssim, calculate_sam, 
    calculate_ergas, calculate_cc, calculate_rmse, calculate_uiqi
)

def main():
    # ================== 实验配置 ==================
    config_path = "configs/models/unsupervised_dynamic_pan.yaml"
    checkpoint_path = "work_dirs/unsupervised_dynamic_pan/best.pth" 
    # ==============================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Evaluation on {device}")

    # 1. 加载配置、模型与数据
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model = MODELS.build(cfg['model']).to(device)
    val_dataset = DATASETS.build(cfg['dataset_test'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 2. 加载预训练权重
    if os.path.exists(checkpoint_path):
        print(f"📦 Loading Best Checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        raise FileNotFoundError(f"❌ Checkpoint not found at {checkpoint_path}. Please train the model first.")

    # 3. 初始化包含 7 大核心指标的字典
    metrics_sum = {
        'PSNR': 0.0, 'SSIM': 0.0, 'SAM': 0.0, 
        'ERGAS': 0.0, 'CC': 0.0, 'RMSE': 0.0, 'UIQI': 0.0
    }
    num_samples = 0

    print("📊 Starting full dataset evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Testing"):
            ms = batch['ms'].to(device)
            pan = batch['pan'].to(device)
            
            if 'gt' not in batch:
                raise ValueError("❌ Ground Truth ('gt') is required to compute metrics.")
            gt = batch['gt'].to(device)

            scale_factor = pan.shape[-1] // ms.shape[-1]

            # 模型推理并限制数值范围
            pred_hrms = model(ms, pan)
            pred_hrms = torch.clamp(pred_hrms, 0, 1)
            gt = torch.clamp(gt, 0, 1)

            # 转为 Numpy (H, W, C) 用于科学计算
            pred_np = pred_hrms.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            gt_np = gt.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            # 调用解耦的指标计算函数
            metrics_sum['PSNR'] += calculate_psnr(pred_np, gt_np)
            metrics_sum['SSIM'] += calculate_ssim(pred_np, gt_np)
            metrics_sum['SAM'] += calculate_sam(pred_np, gt_np)
            metrics_sum['ERGAS'] += calculate_ergas(pred_np, gt_np, scale_factor)
            metrics_sum['CC'] += calculate_cc(pred_np, gt_np)
            metrics_sum['RMSE'] += calculate_rmse(pred_np, gt_np)
            metrics_sum['UIQI'] += calculate_uiqi(pred_np, gt_np)
            
            num_samples += 1

    # 4. 打印最终的霸气表格
    print("\n" + "="*55)
    print(f"🏆 Final Results on {len(val_dataset)} Samples 🏆")
    print("="*55)
    print(f" 📈 PSNR  (越大越好) : {metrics_sum['PSNR'] / num_samples:.4f} dB")
    print(f" 📈 SSIM  (越大越好) : {metrics_sum['SSIM'] / num_samples:.4f}")
    print(f" 📉 SAM   (越小越好) : {metrics_sum['SAM'] / num_samples:.4f} °")
    print(f" 📉 ERGAS (越小越好) : {metrics_sum['ERGAS'] / num_samples:.4f}")
    print(f" 📈 CC    (越大越好) : {metrics_sum['CC'] / num_samples:.4f}")
    print(f" 📉 RMSE  (越小越好) : {metrics_sum['RMSE'] / num_samples:.4f}")
    print(f" 📈 UIQI  (越大越好) : {metrics_sum['UIQI'] / num_samples:.4f}")
    print("="*55 + "\n")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()