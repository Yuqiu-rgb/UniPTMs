class HierarchicalDynamicWeightedFusion(nn.Module):
    def __init__(self, dim=1024, num_heads=8, expansion_ratio=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.expansion_dim = dim * expansion_ratio
        
        # 主分支增强投影
        self.master_proj = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, 3*self.expansion_dim),
            ChannelWeighting(dim=3*self.expansion_dim)  # 新增通道加权
        )
        
        # 从分支约束投影
        self.slave_proj = nn.Sequential(
            nn.Linear(dim, 2*dim),
            SpatialWeighting(dim=2*dim),  # 新增空间加权
            nn.GELU(),
            nn.Linear(2*dim, self.expansion_dim)
        )
        
        # 三级加权融合系统
        self.weight_system = nn.ModuleDict({
            'attention_weight': AttentionWeighting(dim),
            'feature_weight': FeatureWeighting(dim, num_heads),
            'residual_weight': ResidualWeighting(dim)
        })
        
        # 稳定性增强
        self.layer_scale = nn.Parameter(torch.ones(1, 1, dim)*1e-6)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, master, slave):
        # 阶段1：主分支特征强化
        m_qkv = self.master_proj(master)  # 带通道加权
        m_q, m_k, m_v = rearrange(m_qkv, 'b l (n e) -> n b l e', n=3)
        
        # 阶段2：从分支特征调制
        s_ctx = self.slave_proj(slave)  # 带空间加权
        
        # 阶段3：三级加权融合
        # (1) 注意力加权
        attn_logits = einsum(m_q, s_ctx, 'b l e, b s e -> b l s')
        attn_weights = self.weight_system['attention_weight'](attn_logits, master)
        
        # (2) 特征聚合加权
        gathered = einsum(attn_weights, m_v, 'b l s, b s e -> b l e')
        gathered = self.weight_system['feature_weight'](gathered, master)
        
        # (3) 残差加权融合
        fused = self.weight_system['residual_weight'](gathered, master)
        
        # 稳定性增强
        return self.norm(fused + self.layer_scale * master)

# 新增核心组件
class ChannelWeighting(nn.Module):
    """主分支通道重要性加权"""
    def __init__(self, dim):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.channel_gate(x.mean(dim=1, keepdim=True))

class SpatialWeighting(nn.Module):
    """从分支空间位置加权"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, groups=8)
        self.gate = nn.Sequential(
            nn.Conv1d(dim, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        spatial_feat = self.conv(x.transpose(1,2)).transpose(1,2)
        return x * self.gate(spatial_feat)

class AttentionWeighting(nn.Module):
    """注意力得分动态加权"""
    def __init__(self, dim):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)*0.5)
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, attn_logits, master):
        # 主分支引导的温度调节
        temp = self.temperature + master.mean(dim=-1).unsqueeze(-1)
        attn = F.softmax(attn_logits * temp, dim=-1)
        return self.alpha * attn

class FeatureWeighting(nn.Module):
    """特征聚合加权"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, num_heads),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features, master):
        weights = self.mlp(master).transpose(1,2)  # [B, H, L]
        return einsum(weights, features, 'b h l, b l e -> b h e').flatten(1,2)

class ResidualWeighting(nn.Module):
    """残差连接动态加权"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        
    def forward(self, fused, master):
        return (1 + self.gamma) * fused + self.beta * master