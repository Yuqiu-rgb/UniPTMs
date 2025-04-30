import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv1D(nn.Module):
    """动态卷积核生成器"""
    def __init__(self, d_model=1024, kernel_size=3):
        super().__init__()
        self.kernel_gen = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model * kernel_size)
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        """输入x: [B, L, D] 输出: [B, L, D]"""
        B, L, D = x.shape
        # 生成动态卷积核参数
        kernel_params = self.kernel_gen(x.mean(dim=1))  # [B, D*K]
        kernel = kernel_params.view(B, D, self.kernel_size)  # [B, D, K]
        
        # 分组卷积实现
        x = x.transpose(1, 2)  # [B, D, L]
        x = F.conv1d(x, kernel, padding='same', groups=D)
        return x.transpose(1, 2)  # [B, L, D]

class BHGFN(nn.Module):
    def __init__(self, d_model=1024, d_aux=256, n_heads=8):  #d_aux can be changed (up to stage,it can be 256,512,1024,here 256 is a test dim for readers)
        super().__init__()
        # 第一阶段：主->辅助特征交互
        self.cross_attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # 第二阶段：辅助->主特征交互
        self.cross_attn2 = nn.MultiheadAttention(d_aux, n_heads, batch_first=True)
        
        # 动态卷积增强
        self.dynamic_conv = DynamicConv1D(d_model)
        
        # 三级门控机制
        self.attn_gate = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.Sigmoid()
        )
        self.channel_gate = nn.Linear(d_model, d_model)
        self.spatial_gate = nn.Conv1d(d_model, 1, kernel_size=1)
        
        # 维度匹配
        self.aux_proj = nn.Linear(d_aux, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H1, H2):
        # ===== 第一阶段：主特征引导的交叉注意力 =====
        # 主特征作为Query，辅助特征作为Key/Value
        attn_out1, _ = self.cross_attn1(
            query=H1,
            key=self.aux_proj(H2),
            value=self.aux_proj(H2)
        )
        
        # ===== 第二阶段：辅助特征反向注意力 =====
        # 辅助特征作为Query，主特征作为Key/Value
        attn_out2, _ = self.cross_attn2(
            query=H2,
            key=H1,
            value=H1
        )
        attn_out2 = self.aux_proj(attn_out2)  # 投影到主特征空间
        
        # ===== 注意力级门控融合 =====
        combined = torch.cat([attn_out1, attn_out2], dim=-1)
        alpha = self.attn_gate(combined)  # [B, L, D]
        fused = alpha * attn_out1 + (1-alpha) * attn_out2
        
        # ===== 动态卷积增强 =====
        conv_out = self.dynamic_conv(fused)
        
        # ===== 通道级门控 =====
        channel_weights = torch.sigmoid(self.channel_gate(H1))
        gated_conv = conv_out * channel_weights
        
        # ===== 空间级门控 =====
        spatial_weights = torch.sigmoid(self.spatial_gate(gated_conv.transpose(1,2)))
        final_out = gated_conv * spatial_weights.transpose(1,2)
        
        # ===== 残差连接 =====
        return self.norm(final_out + H1)

