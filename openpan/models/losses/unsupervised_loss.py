import torch
import torch.nn as nn
import torch.nn.functional as F
from openpan.registry import LOSSES

@LOSSES.register_module()
class UnsupervisedPanLoss(nn.Module):
    def __init__(self, spatial_weight=1.0, spectral_weight=1.0):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.spectral_weight = spectral_weight

    def spatial_gradient(self, x):
        """提取图像的梯度（边缘特征）代表高频空间信息"""
        # 简单的水平和垂直差分
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x-1, :])
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x-1])
        return h_tv, w_tv

    def forward(self, pred_hrms, ms_lr, pan_hr):
        """
        pred_hrms: 网络预测的高分辨率多光谱
        ms_lr: 原始低分辨率多光谱 (输入)
        pan_hr: 原始高分辨率全色 (输入)
        """
        # ==========================================
        # 1. 光谱一致性损失 (Spectral Consistency)
        # 模拟传感器成像：HRMS 退化后必须等于 LRMS
        # 这里用简单的 AvgPool 模拟空间下采样退化 (实际可用更严谨的高斯PSF)
        # ==========================================
        scale_factor = pan_hr.shape[-1] // ms_lr.shape[-1]
        pred_hrms_down = F.avg_pool2d(pred_hrms, kernel_size=scale_factor, stride=scale_factor)
        loss_spectral = F.l1_loss(pred_hrms_down, ms_lr)

        # ==========================================
        # 2. 空间一致性损失 (Spatial Consistency)
        # HRMS 的空间梯度应该与 PAN 的空间梯度相似
        # ==========================================
        pred_h, pred_w = self.spatial_gradient(pred_hrms)
        pan_h, pan_w = self.spatial_gradient(pan_hr)
        
        # 将 PAN 的梯度广播到与 MS 相同的波段数进行对比
        pan_h = pan_h.expand_as(pred_h)
        pan_w = pan_w.expand_as(pred_w)
        
        loss_spatial = F.l1_loss(pred_h, pan_h) + F.l1_loss(pred_w, pan_w)

        # 总体无监督 Loss
        total_loss = self.spectral_weight * loss_spectral + self.spatial_weight * loss_spatial
        
        return total_loss, {"loss_spectral": loss_spectral.item(), "loss_spatial": loss_spatial.item()}