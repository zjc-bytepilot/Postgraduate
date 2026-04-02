import torch
import torch.nn as nn
import torch.nn.functional as F
from openpan.registry import BACKBONES

class SceneStatisticExtractor(nn.Module):
    """
    场景统计特征提取器：计算能代表当前数据的各种分布特征
    提取的特征固定长度，与波段数无关，增强泛化能力。
    """
    def __init__(self):
        super().__init__()
        
    def _get_spatial_gradient_energy(self, x):
        """计算空间梯度能量 (代表图像高频丰富度)"""
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x-1, :], 2).mean(dim=[2,3])
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x-1], 2).mean(dim=[2,3])
        return h_tv + w_tv

    def forward(self, ms, pan):
        B, C, H, W = ms.shape
        
        # 1. PAN的统计特征 (尺寸不变)
        pan_mean = pan.mean(dim=[2,3]) # (B, 1)
        pan_var = pan.var(dim=[2,3])   # (B, 1)
        pan_grad = self._get_spatial_gradient_energy(pan) # (B, 1)
        
        # 2. MS的统计特征 (为了应对波段数C变化，我们在通道维度也求平均)
        ms_mean = ms.mean(dim=[1,2,3], keepdim=True).squeeze(-1).squeeze(-1) # (B, 1)
        ms_var = ms.var(dim=[2,3]).mean(dim=1, keepdim=True)                 # (B, 1)
        ms_grad = self._get_spatial_gradient_energy(ms).mean(dim=1, keepdim=True) # (B, 1)
        
        # 3. 拼接成一个固定长度的向量 (此处长度为 6)
        scene_stats = torch.cat([pan_mean, pan_var, pan_grad, ms_mean, ms_var, ms_grad], dim=1)
        
        return scene_stats # (B, 6)

class StatDynamicConv2d(nn.Module):
    """基于数据统计特征驱动的动态卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_experts=4, stat_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts

        # 1. 定义 K 个固定的"专家卷积核"集合
        self.weight_experts = nn.Parameter(
            torch.randn(num_experts, out_channels, in_channels, kernel_size, kernel_size) * 0.02
        )
        self.bias_experts = nn.Parameter(torch.zeros(num_experts, out_channels))

        # 2. 路由网络
        self.router = nn.Sequential(
            nn.Linear(stat_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_experts),
            nn.Softmax(dim=1) 
        )

    def forward(self, x, scene_stats):
        B = x.shape[0]

        # 1. 计算专家权重
        routing_weights = self.router(scene_stats) 

        # 2. 动态融合出当前图像专属的卷积核
        dynamic_weight = torch.einsum('bk,koihw->boihw', routing_weights, self.weight_experts)
        dynamic_bias = torch.einsum('bk,ko->bo', routing_weights, self.bias_experts)

        # 3. 执行 Group Conv
        x_reshaped = x.view(1, B * self.in_channels, x.shape[2], x.shape[3])
        dynamic_weight_reshaped = dynamic_weight.view(
            B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        
        out = F.conv2d(
            x_reshaped, weight=dynamic_weight_reshaped, bias=None, 
            stride=self.stride, padding=self.padding, groups=B
        )
        
        out = out.view(B, self.out_channels, out.shape[2], out.shape[3])
        out = out + dynamic_bias.view(B, self.out_channels, 1, 1)

        return out

class DynamicModulationBlock(nn.Module):
    def __init__(self, channels, stat_dim=6):
        super().__init__()
        self.dynamic_conv = StatDynamicConv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, num_experts=4, stat_dim=stat_dim
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, scene_stats):
        feat = self.dynamic_conv(x, scene_stats)
        return self.act(feat)

@BACKBONES.register_module()
class DynamicAgnosticBackbone(nn.Module):
    """波段不可知的动态主干网络"""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.stat_extractor = SceneStatisticExtractor()
        
        # 共享单通道提取器
        self.shared_proj = nn.Conv2d(1, embed_dim, 3, 1, 1)
        
        # 动态特征处理层
        self.ms_mod_block = DynamicModulationBlock(embed_dim, stat_dim=6)
        self.pan_mod_block = DynamicModulationBlock(embed_dim, stat_dim=6)

    def forward(self, ms_up, pan):
        B, C, H, W = ms_up.shape
        
        # 1. 提取先验
        scene_stats = self.stat_extractor(ms_up, pan) 
        
        # 2. 波段不可知提取 (Band-Agnostic)
        ms_flat = ms_up.view(B * C, 1, H, W)
        ms_feat_flat = self.shared_proj(ms_flat)
        pan_feat = self.shared_proj(pan) 
        
        # 融合成 (B, embed_dim, H, W)
        ms_feat = ms_feat_flat.view(B, C, self.embed_dim, H, W).mean(dim=1) 
        
        # 3. 动态调制
        ms_feat_dynamic = self.ms_mod_block(ms_feat, scene_stats)
        pan_feat_dynamic = self.pan_mod_block(pan_feat, scene_stats)
        
        return ms_feat_dynamic, pan_feat_dynamic