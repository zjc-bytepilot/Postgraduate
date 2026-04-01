import torch
import torch.nn as nn
from openpan.registry import NECKS

@NECKS.register_module()
class SpectralGuidedInjectionNeck(nn.Module):
    """空间特征变换 (SFT) 颈部，实现高泛化性的特征注入"""
    def __init__(self, embed_dim=64):
        super().__init__()
        # 用 MS 特征去生成调节 PAN 特征的缩放因子(gamma)和偏置项(beta)
        self.cond_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(embed_dim, embed_dim * 2, 3, 1, 1) # 生成两倍通道，一半是gamma，一半是beta
        )

    def forward(self, ms_feat, pan_feat):
        # 1. 基于 MS 现状生成注入规律
        cond = self.cond_conv(ms_feat)
        gamma, beta = torch.chunk(cond, 2, dim=1)
        
        # 2. 调制 PAN 的高频特征 (SFT 变换)
        # 这在物理上等价于根据不同区域的光谱反射率，动态决定注入多少纹理
        injected_pan = pan_feat * (gamma + 1.0) + beta
        
        # 3. 融合并输出
        fused_feat = ms_feat + injected_pan
        return fused_feat