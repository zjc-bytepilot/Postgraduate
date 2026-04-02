import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpan.registry import LOSSES

class MTFDownsampler(nn.Module):
    """
    基于卫星传感器 MTF (调制传递函数) 的物理退化降采样器。
    先进行高斯低通滤波，再进行间隔抽样，完美替代不合理的 AvgPool。
    """
    def __init__(self, kernel_size=11, sigma=1.5):
        super().__init__()
        self.kernel_size = kernel_size
        # 生成二维高斯核 (模拟 PSF)
        kernel = self._get_gaussian_kernel(kernel_size, sigma)
        # 注册为 buffer，这样它会被移动到 GPU，但不会被优化器更新
        self.register_buffer('gaussian_kernel', kernel)

    def _get_gaussian_kernel(self, kernel_size, sigma):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        
        # 计算 2D 高斯公式
        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )
        # 归一化，保证能量守恒
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        # 形状转为 (1, 1, K, K) 以便后续深度可分离卷积使用
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, x, scale_factor):
        B, C, H, W = x.shape
        
        # 1. 动态复制高斯核以匹配输入波段数 C (波段不可知特性的核心)
        weight = self.gaussian_kernel.repeat(C, 1, 1, 1)
        
        # 2. 边缘反射填充 (防止卷积后边缘变黑)
        pad = self.kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        
        # 3. 深度可分离卷积 (Depthwise Conv) 进行高斯模糊，各个波段独立滤波
        x_blur = F.conv2d(x_pad, weight, groups=C)
        
        # 4. 间隔抽样 (Decimation) 实现降尺度
        x_down = x_blur[:, :, ::scale_factor, ::scale_factor]
        
        return x_down

@LOSSES.register_module()
class UnsupervisedPanLoss(nn.Module):
    def __init__(self, spatial_weight=1.0, spectral_weight=0.5, sigma=1.5):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.spectral_weight = spectral_weight
        
        # 实例化物理退化器 (sigma 根据经验通常设为 1.5 到 2.0 之间)
        self.mtf_downsampler = MTFDownsampler(kernel_size=11, sigma=sigma)

    def spatial_gradient(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x-1, :])
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x-1])
        return h_tv, w_tv

    def forward(self, pred_hrms, ms_lr, pan_hr):
        # 计算尺度因子 (例如 GF-2 是 4，WV-3 通常也是 4)
        scale_factor = pan_hr.shape[-1] // ms_lr.shape[-1]
        
        # ==========================================
        # 🌟 1. 光谱一致性 (Wald's protocol 物理升级版)
        # ==========================================
        # 摒弃 F.avg_pool2d，采用物理 MTF 降采样
        pred_hrms_down = self.mtf_downsampler(pred_hrms, scale_factor)
        loss_spectral = F.l1_loss(pred_hrms_down, ms_lr)

        # ==========================================
        # 2. 空间一致性 (Gradient correlation)
        # ==========================================
        pred_h, pred_w = self.spatial_gradient(pred_hrms)
        pan_h, pan_w = self.spatial_gradient(pan_hr)
        
        pan_h = pan_h.expand_as(pred_h)
        pan_w = pan_w.expand_as(pred_w)
        
        loss_spatial = F.l1_loss(pred_h, pan_h) + F.l1_loss(pred_w, pan_w)

        total_loss = self.spectral_weight * loss_spectral + self.spatial_weight * loss_spatial
        
        return total_loss, {"loss_spectral": loss_spectral.item(), "loss_spatial": loss_spatial.item()}