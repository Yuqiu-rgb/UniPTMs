import torch
import torch.nn as nn
import torch.nn.functional as F
"""
@author: linyiyu
this file is under testing
"""
class LDFusion(nn.Module):
    def __init__(self, dim1=164, dim2=30, hidden_dim=256, output_dim=512): # dim1,dim2,output_dim can be changed for you need.
        super().__init__()
        
        # 维度对齐投影
        self.proj1 = nn.Sequential(
            nn.Linear(dim1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(dim2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 交叉密集注意力模块
        self.cross_attn = nn.ModuleDict({
            'ember2q': nn.Linear(hidden_dim, hidden_dim),
            'pseaacq': nn.Linear(hidden_dim, hidden_dim),
            'kv': nn.Linear(hidden_dim*2, 2*hidden_dim)
        })
        
        # 动态特征蒸馏
        self.distill_gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 多尺度特征提取
        self.conv_branch = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        self.conv_fuse = nn.Linear(3*hidden_dim, hidden_dim)
        
        # 自适应融合
        self.fusion_layer = nn.Linear(hidden_dim*3, output_dim)
        
        # 初始化
        nn.init.kaiming_normal_(self.proj1[0].weight, mode='fan_out', nonlinearity='gelu')
        nn.init.kaiming_normal_(self.proj2[0].weight, mode='fan_out', nonlinearity='gelu')

    def forward(self, x1, x2):
        """
        x1: EMBER2嵌入 
        x2: PseAAC嵌入 
        """
        # 维度对齐
        h1 = self.proj1(x1)  # [B, L, D]
        h2 = self.proj2(x2)  # [B, L, D]
        
        # 交叉注意力计算
        Q1 = self.cross_attn['ember2q'](h1)  # EMBER2作为Query
        Q2 = self.cross_attn['pseaacq'](h2)  # PseAAC作为Query
        KV = self.cross_attn['kv'](torch.cat([h1, h2], dim=-1)).chunk(2, dim=-1)
        
        # 双向注意力
        attn1 = F.scaled_dot_product_attention(Q1, *KV)  # EMBER2引导的注意力
        attn2 = F.scaled_dot_product_attention(Q2, *KV)  # PseAAC引导的注意力
        
        # 动态特征蒸馏
        gate = self.distill_gate(torch.cat([attn1, attn2], dim=-1))
        distilled = gate * attn1 + (1 - gate) * attn2
        
        # 多尺度卷积增强
        conv_feats = []
        for conv in self.conv_branch:
            feat = conv(distilled.transpose(1,2)).transpose(1,2)
            conv_feats.append(feat)
        fused_conv = self.conv_fuse(torch.cat(conv_feats, dim=-1))
        
        # 最终融合
        combined = torch.cat([distilled, fused_conv, h1 + h2], dim=-1)
        output = self.fusion_layer(combined)
        
        return output

