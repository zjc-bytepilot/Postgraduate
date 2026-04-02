import torch
import torch.nn as nn
from openpan.registry import HEADS

@HEADS.register_module()
class DynamicAgnosticHead(nn.Module):
    """波段不可知的动态重建头"""
    def __init__(self, embed_dim=64):
        super().__init__()
        # 仅预测 1 个通道的通用高频细节
        self.detail_predictor = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // 2, 1, 3, 1, 1) 
        )

    def forward(self, fused_feat, ms_up):
        base_details = self.detail_predictor(fused_feat) # (B, 1, H, W)
        
        # 提取 MS 能量分布用于动态分配权重
        B, C, H, W = ms_up.shape
        band_energy = ms_up.mean(dim=[2,3], keepdim=True) 
        band_weights = band_energy / (band_energy.mean(dim=1, keepdim=True) + 1e-8)
        
        # 广播单通道细节到 C 个通道
        dynamic_details = base_details * band_weights
        
        return dynamic_details