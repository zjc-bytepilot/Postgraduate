import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img_fake, img_real):
    """PSNR: 峰值信噪比 (越大越好)"""
    mse = np.mean((img_fake - img_real) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)

def calculate_ssim(img_fake, img_real):
    """SSIM: 结构相似度 (越大越好)"""
    return ssim(img_fake, img_real, data_range=1.0, channel_axis=-1)

def calculate_sam(img_fake, img_real):
    """SAM: 光谱角映射 (越小越好)"""
    fake_flat = img_fake.reshape(-1, img_fake.shape[-1])
    real_flat = img_real.reshape(-1, img_real.shape[-1])
    
    dot_product = np.sum(fake_flat * real_flat, axis=1)
    norm_fake = np.linalg.norm(fake_flat, axis=1)
    norm_real = np.linalg.norm(real_flat, axis=1)
    
    val = dot_product / (norm_fake * norm_real + 1e-8)
    val = np.clip(val, -1.0, 1.0) 
    
    sam_angles = np.arccos(val)
    return np.mean(sam_angles) * 180 / np.pi 

def calculate_ergas(img_fake, img_real, scale):
    """ERGAS: 综合相对无量纲全局误差 (越小越好)"""
    channels = img_fake.shape[-1]
    inner_sum = 0.0
    for i in range(channels):
        band_fake = img_fake[:, :, i]
        band_real = img_real[:, :, i]
        rmse_sq = np.mean((band_fake - band_real) ** 2)
        mean_real = np.mean(band_real)
        if mean_real != 0:
            inner_sum += (rmse_sq / (mean_real ** 2))
            
    ergas = 100.0 / scale * np.sqrt(inner_sum / channels)
    return ergas

def calculate_cc(img_fake, img_real):
    """CC: 空间相关系数 (越大越好)"""
    channels = img_fake.shape[-1]
    cc_sum = 0.0
    for i in range(channels):
        band_fake = img_fake[:, :, i]
        band_real = img_real[:, :, i]
        
        mean_fake = np.mean(band_fake)
        mean_real = np.mean(band_real)
        
        cov = np.mean((band_fake - mean_fake) * (band_real - mean_real))
        std_fake = np.std(band_fake)
        std_real = np.std(band_real)
        
        if std_fake * std_real != 0:
            cc_sum += cov / (std_fake * std_real)
            
    return cc_sum / channels

def calculate_rmse(img_fake, img_real):
    """RMSE: 均方根误差 (越小越好)"""
    return np.sqrt(np.mean((img_fake - img_real) ** 2))

def calculate_uiqi(img_fake, img_real):
    """
    UIQI: 通用图像质量指数 (Universal Image Quality Index, 越大越好)
    由三个分量组成：相关度失真、亮度失真、对比度失真。
    此处计算各个波段的全局 UIQI 并求平均。
    """
    channels = img_fake.shape[-1]
    q_sum = 0.0
    for i in range(channels):
        x = img_fake[:, :, i]
        y = img_real[:, :, i]
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        var_x = np.var(x)
        var_y = np.var(y)
        cov_xy = np.mean((x - mean_x) * (y - mean_y))
        
        numerator = 4 * cov_xy * mean_x * mean_y
        denominator = (var_x + var_y) * (mean_x**2 + mean_y**2)
        
        if denominator != 0:
            q_sum += numerator / denominator
            
    return q_sum / channels