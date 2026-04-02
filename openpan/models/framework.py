import torch.nn as nn
import torch.nn.functional as F
from openpan.registry import MODELS, BACKBONES, NECKS, HEADS

@MODELS.register_module()
class UnsupervisedPanModel(nn.Module):
    """高度解耦的 Pansharpening 深度学习通用框架"""
    def __init__(self, backbone_cfg, neck_cfg, head_cfg):
        super().__init__()
        self.backbone = BACKBONES.build(backbone_cfg)
        self.neck = NECKS.build(neck_cfg)
        self.head = HEADS.build(head_cfg)

    def forward(self, ms, pan):
        # 1. 空间上采样
        ms_up = F.interpolate(ms, size=pan.shape[-2:], mode='bicubic', align_corners=False)
        
        # 2. 波段不可知与动态提取
        ms_feat, pan_feat = self.backbone(ms_up, pan)
        
        # 3. SFT 融合
        fused_feat = self.neck(ms_feat, pan_feat)
        
        # 4. 动态能量重组
        details = self.head(fused_feat, ms_up)
        
        # 5. 光谱残差保真
        out_hrms = ms_up + details
        return out_hrms