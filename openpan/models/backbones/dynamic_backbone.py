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
        
        # 2. MS的统计特征 (为了应对波段数C变化，我们在通道维度也求平均或最大值)
        ms_mean = ms.mean(dim=[1,2,3], keepdim=True).squeeze(-1).squeeze(-1) # (B, 1)
        ms_var = ms.var(dim=[2,3]).mean(dim=1, keepdim=True)                 # (B, 1)
        ms_grad = self._get_spatial_gradient_energy(ms).mean(dim=1, keepdim=True) # (B, 1)
        
        # 3. 拼接成一个固定长度的向量 (此处长度为 6)
        # 这个向量包含了当前输入图像的亮度、对比度、纹理丰富度的全局先验！
        scene_stats = torch.cat([pan_mean, pan_var, pan_grad, ms_mean, ms_var, ms_grad], dim=1)
        
        return scene_stats # (B, 6)

class StatDynamicConv2d(nn.Module):
    """
    基于数据统计特征驱动的动态卷积层
    它的卷积核权重不是固定的，而是根据当前输入数据的分布实时生成的！
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_experts=4, stat_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts

        # 1. 定义 K 个固定的"专家卷积核"集合 (Parameter)
        # 形状: (num_experts, out_channels, in_channels, kernel_size, kernel_size)
        self.weight_experts = nn.Parameter(
            torch.randn(num_experts, out_channels, in_channels, kernel_size, kernel_size) * 0.02
        )
        self.bias_experts = nn.Parameter(torch.zeros(num_experts, out_channels))

        # 2. 路由网络 (Router): 接收数据分布特征，决定用哪些专家的知识
        self.router = nn.Sequential(
            nn.Linear(stat_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_experts),
            nn.Softmax(dim=1) # 保证 K 个专家的权重和为 1
        )

    def forward(self, x, scene_stats):
        """
        x: 输入图像/特征图 (B, in_channels, H, W)
        scene_stats: 当前数据的分布特征 (B, stat_dim)
        """
        B = x.shape[0]

        # 1. 根据当前数据的分布，计算 K 个专家的注意力权重
        # routing_weights 形状: (B, num_experts)
        routing_weights = self.router(scene_stats) 

        # 2. 🌟 见证奇迹的时刻：动态融合出当前图像专属的卷积核参数！
        # 针对 Batch 中的每一张图，都会生成一个完全不同的卷积核
        # 形状变换: (B, K) * (K, C_out, C_in, K_h, K_w) -> (B, C_out, C_in, K_h, K_w)
        dynamic_weight = torch.einsum('bk,koihw->boihw', routing_weights, self.weight_experts)
        dynamic_bias = torch.einsum('bk,ko->bo', routing_weights, self.bias_experts)

        # 3. 使用生成的动态卷积核去处理图像
        # 因为不同 batch 的卷积核不同，需要将 batch 维合并到通道维进行 Group Conv
        x_reshaped = x.view(1, B * self.in_channels, x.shape[2], x.shape[3])
        dynamic_weight_reshaped = dynamic_weight.view(
            B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        
        out = F.conv2d(
            x_reshaped, 
            weight=dynamic_weight_reshaped, 
            bias=None, 
            stride=self.stride, 
            padding=self.padding, 
            groups=B
        )
        
        # 恢复形状并加上动态偏置
        out = out.view(B, self.out_channels, out.shape[2], out.shape[3])
        out = out + dynamic_bias.view(B, self.out_channels, 1, 1)

        return out
    
class DynamicModulationBlock(nn.Module):
    def __init__(self, channels, stat_dim=6):
        super().__init__()
        # 🌟 直接使用动态卷积替换普通卷积
        # 这个卷积层不再有固定的参数，每次前向传播它的参数都在变！
        self.dynamic_conv = StatDynamicConv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=3, 
            num_experts=4, # 设定4个专家
            stat_dim=stat_dim
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, scene_stats):
        # 将数据分布特征传入，动态生成卷积核并执行卷积
        feat = self.dynamic_conv(x, scene_stats)
        return self.act(feat)

@BACKBONES.register_module()
class DynamicAgnosticBackbone(nn.Module):
    """
    波段不可知的动态主干网络：
    1. 支持任意波段数输入。
    2. 基于数据分布特征动态调整网络权重。
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 统计特征提取器
        self.stat_extractor = SceneStatisticExtractor()
        
        # 共享的单通道特征提取器 (用于把MS的任意波段和PAN都映射到同一个隐空间)
        self.shared_proj = nn.Conv2d(1, embed_dim, 3, 1, 1)
        
        # 动态调制层
        self.ms_mod_block = DynamicModulationBlock(embed_dim, stat_dim=6)
        self.pan_mod_block = DynamicModulationBlock(embed_dim, stat_dim=6)

    def forward(self, ms_up, pan):
        B, C, H, W = ms_up.shape
        
        # ===============================================
        # 步骤 1：提取当前数据的全局分布先验
        # ===============================================
        scene_stats = self.stat_extractor(ms_up, pan) # (B, 6)
        
        # ===============================================
        # 步骤 2：波段不可知的特征提取 (Band-Agnostic)
        # ===============================================
        # 将 ms_up 从 (B, C, H, W) 变成 (B*C, 1, H, W)
        ms_flat = ms_up.view(B * C, 1, H, W)
        
        # 用共享卷积核提取特征，输出 (B*C, embed_dim, H, W)
        ms_feat_flat = self.shared_proj(ms_flat)
        pan_feat = self.shared_proj(pan) # (B, embed_dim, H, W)
        
        # 变回 (B, C, embed_dim, H, W)，然后把波段维度的特征融合掉 (比如取平均)
        ms_feat = ms_feat_flat.view(B, C, self.embed_dim, H, W).mean(dim=1) # (B, embed_dim, H, W)
        
        # ===============================================
        # 步骤 3：数据驱动的动态调制 (Data-Driven Modulation)
        # ===============================================
        # 此时的特征提取不再是死板的卷积，而是受到场景统计特征影响的
        ms_feat_dynamic = self.ms_mod_block(ms_feat, scene_stats)
        pan_feat_dynamic = self.pan_mod_block(pan_feat, scene_stats)
        
        return ms_feat_dynamic, pan_feat_dynamic