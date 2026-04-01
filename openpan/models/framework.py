import torch
import torch.nn as nn
import torch.nn.functional as F
from openpan.registry import MODELS, BACKBONES, NECKS, HEADS

@MODELS.register_module()
class UnsupervisedPanModel(nn.Module):
    def __init__(self, backbone_cfg, neck_cfg, head_cfg):
        super().__init__()
        self.backbone = BACKBONES.build(backbone_cfg)
        self.neck = NECKS.build(neck_cfg)
        self.head = HEADS.build(head_cfg)

    def forward(self, ms, pan):
        # 1. 空间上采样 MS 匹配 PAN 尺寸
        ms_up = F.interpolate(ms, size=pan.shape[-2:], mode='bicubic', align_corners=False)
        
        # 2. 提取自适应特征
        ms_feat, pan_feat = self.backbone(ms_up, pan)
        
        # 3. 光谱引导的动态注入
        fused_feat = self.neck(ms_feat, pan_feat)
        
        # 4. 预测高频残差细节 (Details)
        details = self.head(fused_feat)
        
        # 5. 绝对光谱保真机制：原始上采样MS + 预测细节
        out_hrms = ms_up + details
        return out_hrms