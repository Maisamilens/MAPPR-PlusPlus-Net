"""
Multi-path Attention Residual Fusion Plus Plus (MARF++) Block

Reference: Inspired by Multi-path Attention Residual Fusion from
ScienceDirect (2025) - https://www.sciencedirect.com/science/article/pii/S1532046425001947

Implementation follows Equations (13)-(18) of the MAPPR++-Net paper:
  Fα = SA(CA(DWConv(AvgP(X))))           — Channel-Spatial cascade
  Fβ = CNL(AvgP(X))                       — Channel Non-Local
  Fγ = LTA(MSFP(X))                       — Lightweight Transformer-Assisted
  WDPS = Softmax(GAP([Fα; Fβ; Fγ]))       — Dynamic Path Selection
  FCPI = CrossAttn([Fα⊕Fβ; Fβ⊕Fγ; Fγ⊕Fα]) — Cross-Path Interaction
  FMARF++ = Σ W_DPS^(i) ⊙ F_CPI^(i) + X   — Final output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """Channel Attention Module (CA)."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SA)."""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.bn(self.conv(concat)))


class ChannelNonLocal(nn.Module):
    """
    Channel Non-Local (CNL) Attention.
    Captures long-range channel dependencies.
    """
    
    def __init__(self, in_channels: int, reduction: int = 2):
        super(ChannelNonLocal, self).__init__()
        mid = max(1, in_channels // reduction)
        
        self.query = nn.Conv2d(in_channels, mid, 1)
        self.key = nn.Conv2d(in_channels, mid, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.mid = mid
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.query(x).view(B, self.mid, -1)        # [B, D, HW]
        k = self.key(x).view(B, self.mid, -1)           # [B, D, HW]
        v = self.value(x).view(B, C, -1)                # [B, C, HW]
        
        # Channel attention map
        attn = torch.bmm(q.transpose(1, 2), k)          # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.transpose(1, 2))        # [B, C, HW]
        out = out.view(B, C, H, W)
        
        return x + self.gamma * out


class MultiScaleFeaturePyramid(nn.Module):
    """
    Multi-Scale Feature Pyramid (MSFP).
    Captures hierarchical representations at multiple scales.
    """
    
    def __init__(self, in_channels: int, out_channels: int = None):
        super(MultiScaleFeaturePyramid, self).__init__()
        out_channels = out_channels or in_channels
        
        # Scale 1: 3x3
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Scale 2: 3x3 with dilation=2
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2,
                      dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Scale 3: 3x3 with dilation=3
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=3,
                      dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        return self.fuse(torch.cat([s1, s2, s3], dim=1))


class LightweightTransformerAssisted(nn.Module):
    """
    Lightweight Transformer-Assisted (LTA) Path.
    
    Uses efficient multi-head self-attention with MSFP integration.
    """
    
    def __init__(self, in_channels: int, num_heads: int = 4):
        super(LightweightTransformerAssisted, self).__init__()
        
        self.msfp = MultiScaleFeaturePyramid(in_channels)
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = nn.GroupNorm(1, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Apply MSFP first
        x_feat = self.msfp(x)
        
        # Lightweight self-attention
        qkv = self.qkv(x_feat)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, -1)
        k = k.view(B, self.num_heads, self.head_dim, -1)
        v = v.view(B, self.num_heads, self.head_dim, -1)
        
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        
        return self.norm(x_feat + self.gamma * out)


class CrossPathInteraction(nn.Module):
    """
    Cross-Path Interaction (CPI) Module.
    
    Enables information exchange between parallel attention branches
    using lightweight cross-attention.
    
    Args:
        in_channels: Feature channel dimension
        num_heads: Number of attention heads
    """
    
    def __init__(self, in_channels: int, num_heads: int = 4):
        super(CrossPathInteraction, self).__init__()
        self.num_heads = num_heads
        self.head_dim = max(1, in_channels // num_heads)
        self.scale = self.head_dim ** -0.5
        
        # Cross-attention for each pair
        self.q_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = nn.GroupNorm(1, in_channels)
    
    def forward(self, f_alpha, f_beta, f_gamma):
        """
        Cross-path interaction: [Fα⊕Fβ; Fβ⊕Fγ; Fγ⊕Fα]
        """
        # Element-wise sum for cross-path input
        ab = f_alpha + f_beta
        bc = f_beta + f_gamma
        ca = f_gamma + f_alpha
        
        # Cross-attention between pairs
        stack = torch.stack([ab, bc, ca], dim=0)  # [3, B, C, H, W]
        
        B, C, H, W = f_alpha.shape
        
        # Simplified cross-attention
        q = self.q_proj(ab).view(B, C, -1)
        k = self.k_proj(bc).view(B, C, -1)
        v = self.v_proj(ca).view(B, C, -1)
        
        q = q.view(B, self.num_heads, self.head_dim, -1)
        k = k.view(B, self.num_heads, self.head_dim, -1)
        v = v.view(B, self.num_heads, self.head_dim, -1)
        
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.out_proj(out)
        
        return self.norm(out)


class DynamicPathSelection(nn.Module):
    """
    Dynamic Path Selection (DPS) Mechanism.
    
    Adaptively weights different attention pathways based on global
    average pooling of concatenated path features.
    
    WDPS = Softmax(GAP([Fα; Fβ; Fγ]))
    
    Args:
        in_channels: Feature channel dimension
        num_paths: Number of attention paths
    """
    
    def __init__(self, in_channels: int, num_paths: int = 3):
        super(DynamicPathSelection, self).__init__()
        self.num_paths = num_paths
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * num_paths, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_paths),
            nn.Softmax(dim=1)
        )
    
    def forward(self, *path_features):
        """
        Args:
            path_features: Tuple of path features [Fα, Fβ, Fγ]
            
        Returns:
            Weights tensor of shape [B, num_paths, 1, 1]
        """
        B = path_features[0].shape[0]
        
        # Global average pool each path
        pooled = [self.gap(f).view(B, -1) for f in path_features]
        concat = torch.cat(pooled, dim=1)  # [B, C*num_paths]
        
        # Compute weights
        weights = self.fc(concat)  # [B, num_paths]
        weights = weights.view(B, self.num_paths, 1, 1)  # [B, num_paths, 1, 1]
        
        return weights


class MARFPlusPlus(nn.Module):
    """
    Multi-path Attention Residual Fusion Plus Plus (MARF++) Block.
    
    Enhanced version with:
    1. Dynamic Path Selection (DPS) for adaptive path weighting
    2. Cross-Path Interaction (CPI) for inter-branch feature exchange
    3. Multi-Scale Feature Pyramid (MSFP) for hierarchical representation
    
    Three complementary attention pathways:
    - Path α: Channel-Spatial cascade attention
    - Path β: Channel Non-Local attention
    - Path γ: Lightweight Transformer-Assisted with MSFP
    
    Args:
        in_channels: Input feature channels
        num_paths: Number of attention paths (default: 3)
        use_dps: Enable Dynamic Path Selection
        use_cpi: Enable Cross-Path Interaction
        use_msfp: Enable Multi-Scale Feature Pyramid
    """
    
    def __init__(
        self,
        in_channels: int,
        num_paths: int = 3,
        use_dps: bool = True,
        use_cpi: bool = True,
        use_msfp: bool = True,
    ):
        super(MARFPlusPlus, self).__init__()
        
        self.use_dps = use_dps
        self.use_cpi = use_cpi
        
        # Path α: Channel-Spatial cascade attention
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.avg_pool_path = nn.AdaptiveAvgPool2d(1)
        self.path_alpha_ca = ChannelAttention(in_channels)
        self.path_alpha_sa = SpatialAttention()
        
        # Path β: Channel Non-Local attention
        self.path_beta_cnl = ChannelNonLocal(in_channels)
        
        # Path γ: Lightweight Transformer-Assisted with MSFP
        self.path_gamma_lta = LightweightTransformerAssisted(in_channels)
        
        # Dynamic Path Selection
        if use_dps:
            self.dps = DynamicPathSelection(in_channels, num_paths)
        
        # Cross-Path Interaction
        if use_cpi:
            self.cpi = CrossPathInteraction(in_channels)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MARF++ block.
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            Enhanced feature tensor [B, C, H, W]
        """
        identity = x
        
        # Path α: SA(CA(DWConv(AvgP(X)))) (Eq. 13)
        f_alpha = self.dw_conv(x)
        ca_weight = self.path_alpha_ca(f_alpha)
        f_alpha = f_alpha * ca_weight
        sa_weight = self.path_alpha_sa(f_alpha)
        f_alpha = f_alpha * sa_weight
        
        # Path β: CNL(AvgP(X)) (Eq. 14)
        f_beta = self.path_beta_cnl(x)
        
        # Path γ: LTA(MSFP(X)) (Eq. 15)
        f_gamma = self.path_gamma_lta(x)
        
        # Cross-Path Interaction (Eq. 17)
        if self.use_cpi:
            f_cpi = self.cpi(f_alpha, f_beta, f_gamma)
        else:
            f_cpi = (f_alpha + f_beta + f_gamma) / 3.0
        
        # Dynamic Path Selection (Eq. 16) and weighted fusion (Eq. 18)
        if self.use_dps:
            w_dps = self.dps(f_alpha, f_beta, f_gamma)
            # Weighted sum: FMARF++ = Σ W_DPS^(i) ⊙ F_CPI^(i) + X
            f_weighted = w_dps[:, 0] * f_alpha + w_dps[:, 1] * f_beta + w_dps[:, 2] * f_gamma
            out = f_weighted + identity
        else:
            out = f_cpi + identity
        
        out = self.out_proj(out)
        
        return out