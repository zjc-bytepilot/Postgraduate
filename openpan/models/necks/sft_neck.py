import torch
import torch.nn as nn
from openpan.registry import NECKS

@NECKS.register_module()
class SpectralGuidedInjectionNeck(nn.Module):
    """空间特征变换 (SFT) 颈部：用 MS 调制 PAN 细节"""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.cond_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(embed_dim, embed_dim * 2, 3, 1, 1) 
        )

    def forward(self, ms_feat, pan_feat):
        # 生成缩放(gamma)和偏置(beta)
        cond = self.cond_conv(ms_feat)
        gamma, beta = torch.chunk(cond, 2, dim=1)
        
        # SFT 变换
        injected_pan = pan_feat * (gamma + 1.0) + beta
        
        # 融合输出
        fused_feat = ms_feat + injected_pan
        return fused_feat