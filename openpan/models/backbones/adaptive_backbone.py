import torch
import torch.nn as nn
from openpan.registry import BACKBONES

class ChannelSpatialAttention(nn.Module):
    """提取数据自适应的相关性规律 (类似计算半方差/协方差)"""
    def __init__(self, channels):
        super().__init__()
        # Channel Attention (找光谱波段间的相关性)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        # Spatial Attention (找空间像素间的相关性)
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sa_in = torch.cat([max_out, avg_out], dim=1)
        x = x * self.sa(sa_in)
        return x

@BACKBONES.register_module()
class AdaptiveFeatureExtractor(nn.Module):
    def __init__(self, in_ms=4, in_pan=1, embed_dim=64):
        super().__init__()
        self.ms_proj = nn.Conv2d(in_ms, embed_dim, 3, 1, 1)
        self.pan_proj = nn.Conv2d(in_pan, embed_dim, 3, 1, 1)
        
        # 自适应注意力模块
        self.ms_attn = ChannelSpatialAttention(embed_dim)
        self.pan_attn = ChannelSpatialAttention(embed_dim)

    def forward(self, ms_up, pan):
        ms_feat = self.ms_attn(self.ms_proj(ms_up))
        # PAN 特征只关注空间高频规律
        pan_feat = self.pan_attn(self.pan_proj(pan)) 
        return ms_feat, pan_feat