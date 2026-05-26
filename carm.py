"""
Context-Aware Refinement Module (CARM)

Reference: Inspired by context-aware refinement concepts building upon
AGRM (IEEE, 2022) - https://ieeexplore.ieee.org/abstract/document/9624979

Implementation follows Equation (21) of the MAPPR++-Net paper:
  C2 = WConcat(Concat(ASPP(S3), S3), S3)

Captures expansive long-range dependencies within deep feature spaces
using Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Module):
    """Single branch of ASPP with atrous convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super(ASPPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    """Image pooling branch of ASPP."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ASPPPooling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h, w = x.shape[2:]
        pooled = F.adaptive_avg_pool2d(x, 1)
        pooled = self.conv(pooled)
        out = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    
    Reference: DeepLab v3 (Chen et al., 2017)
    https://arxiv.org/abs/1606.00915
    
    Multi-scale feature extraction using parallel atrous convolutions
    with different dilation rates.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels per branch
        rates: List of dilation rates
    """
    
    def __init__(self, in_channels: int, out_channels: int = 256,
                 rates: list = [1, 6, 12, 18]):
        super(ASPP, self).__init__()
        
        branches = []
        
        # 1x1 convolution branch (rate=1)
        branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolution branches
        for rate in rates[1:]:  # Skip rate=1, already covered
            branches.append(ASPPConv(in_channels, out_channels, rate))
        
        # Image pooling branch
        branches.append(ASPPPooling(in_channels, out_channels))
        
        self.branches = nn.ModuleList(branches)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(branches), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        res = [branch(x) for branch in self.branches]
        res = torch.cat(res, dim=1)
        return self.fusion(res)


class CARM(nn.Module):
    """
    Context-Aware Refinement Module.
    
    Captures expansive long-range dependencies within deep feature spaces
    using ASPP for multi-scale context aggregation. Designed for Stage S3
    refinement.
    
    C2 = WConcat(Concat(ASPP(S3), S3), S3)  (Eq. 21)
    
    Args:
        in_channels: Number of input channels (Stage S3)
        out_channels: Number of output channels (for C2)
        aspp_rates: Dilation rates for ASPP
        reduction: Channel reduction for attention
    """
    
    def __init__(self, in_channels: int, out_channels: int = None,
                 aspp_rates: list = [1, 6, 12, 18],
                 reduction: int = 16):
        super(CARM, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        aspp_out = max(1, in_channels // 2)
        
        # ASPP for multi-scale context (Eq. 21: ASPP(S3))
        self.aspp = ASPP(in_channels, aspp_out, aspp_rates)
        
        # Concat(ASPP(S3), S3) projection
        self.concat_proj = nn.Sequential(
            nn.Conv2d(aspp_out + in_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
        # WConcat for final output: WConcat(..., S3) (Eq. 21)
        self.w_concat = nn.Sequential(
            nn.Conv2d(self.out_channels + in_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Optional context attention
        mid_ca = max(1, self.out_channels // reduction)
        self.context_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels, mid_ca, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ca, self.out_channels, 1),
            nn.Sigmoid()
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
    
    def forward(self, s3: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CARM module.
        
        Args:
            s3: Stage 3 features [B, C, H, W]
            
        Returns:
            C2: Refined output features [B, out_channels, H, W]
        """
        # ASPP(S3)
        aspp_out = self.aspp(s3)
        
        # Concat(ASPP(S3), S3) with projection
        concat_out = self.concat_proj(torch.cat([aspp_out, s3], dim=1))
        
        # Context attention weighting
        ca = self.context_attention(concat_out)
        concat_out = concat_out * ca
        
        # C2 = WConcat(Concat(ASPP(S3), S3), S3) (Eq. 21)
        c2 = self.w_concat(torch.cat([concat_out, s3], dim=1))
        
        return c2