import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
"""
@author: linyiyu
this file is under testing
"""
class EnhancedBGCA(nn.Module):
    def __init__(self, dim1=1024, dim2=1280, hidden_dim=512, groups=8, output_dim=1280):
        super().__init__()
        # 分组投影参数
        self.groups = groups
        self.hidden_dim = hidden_dim
        
        # ProtT5投影（分组查询）
        self.proj1_q = nn.Linear(dim1, hidden_dim)
        self.proj1_kv = nn.Sequential(
            nn.Linear(dim1, 2*hidden_dim),
            nn.GELU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim)
        )
        
        # ESM-2投影（分组键值）
        self.proj2_q = nn.Linear(dim2, hidden_dim)
        self.proj2_kv = nn.Sequential(
            nn.Linear(dim2, 2*hidden_dim),
            nn.GELU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim)
        )
        
        # 多尺度动态卷积
        self.dynamic_conv = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2, groups=groups)
            for k in [3, 5, 7]
        ])
        self.conv_gate = nn.Linear(3*hidden_dim, 3)
        
        # 分层门控融合
        self.gate_network = nn.Sequential(
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # 自适应残差
        self.res_adaptor = nn.Linear(dim1+dim2, output_dim)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 输出变换
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, output_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')

    def _group_attention(self, q, k, v):
        """分组注意力计算"""
        # 输入维度: [B, L, H]
        q = rearrange(q, 'b l (g h) -> b g l h', g=self.groups)
        k = rearrange(k, 'b l (g h) -> b g l h', g=self.groups)
        v = rearrange(v, 'b l (g h) -> b g l h', g=self.groups)
        
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0)
        return rearrange(attn, 'b g l h -> b l (g h)')

    def forward(self, x1, x2):
        batch_size, seq_len, _ = x1.shape
        
        # 投影到分组空间
        q1 = self.proj1_q(x1)  # [B, L, H]
        k1, v1 = self.proj1_kv(x1).chunk(2, dim=-1)
        
        q2 = self.proj2_q(x2)  # [B, L, H]
        k2, v2 = self.proj2_kv(x2).chunk(2, dim=-1)
        
        # 双向分组注意力
        attn1 = self._group_attention(q1, k2, v2)  # ProtT5作为query
        attn2 = self._group_attention(q2, k1, v1)  # ESM作为query
        
        # 多尺度动态卷积增强
        conv_feats = []
        for conv in self.dynamic_conv:
            feat = conv(attn1.transpose(1,2)).transpose(1,2)
            conv_feats.append(feat)
        conv_weights = F.softmax(self.conv_gate(torch.cat(conv_feats, dim=-1)), dim=-1)
        attn1 = sum(w.unsqueeze(-1) * f for w, f in zip(conv_weights.unbind(-1), conv_feats))
        
        # 分层门控融合
        combined = torch.stack([attn1, attn2], dim=2)  # [B, L, 2, H]
        gate = self.gate_network(torch.cat([attn1, attn2], dim=-1))  # [B, L, 2]
        fused = (combined * gate.unsqueeze(-1)).sum(dim=2)  # [B, L, H]
        
        # 残差连接与归一化
        fused = self.norm1(fused)
        residual = self.res_adaptor(torch.cat([x1, x2], dim=-1))  # [B, L, D_out]
        output = fused + residual
        
        # 最终输出变换
        return self.norm2(self.output_proj(output))
