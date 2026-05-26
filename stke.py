"""
Shallow Target Knowledge Enhancement (STKE) Module

Reference: Inspired by knowledge enhancement and clinical pathway encoding
from Wu et al. (2024) - "MedKP: Medical Dialogue with Knowledge Enhancement
and Clinical Pathway Encoding"
Paper: https://arxiv.org/pdf/2403.06611

Implementation follows Equations (1)-(4) of the MAPPR++-Net paper:
  Fa = ReLU(Wa * X), Wa ∈ R^{C/4 × C × 1 × 1}
  Fb = ReLU(Wb * X), Wb ∈ R^{C/4 × C × 3 × 3}
  Fc = ReLU(Wc * X), Wc ∈ R^{C/4 × C × 3 × 3}, dilation=2
  Fagg = ReLU(Wf * Concat(Fa, Fb, Fc)), Wf ∈ R^{C/4 × 3C/4 × 1 × 1}
  Ac = σ(W2 * δ(W1 * GAP(Fagg)))
  FSTKE = R(X) + We * (Ac ⊙ Fagg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STKE(nn.Module):
    """
    Shallow Target Knowledge Enhancement Module.
    
    A plug-and-play component that operates after the backbone's first stage (S1).
    Refines low-level features by capturing both fine-grained local structures
    and broader contextual information through multi-scale convolutional streams.
    
    Args:
        in_channels: Number of input feature channels
        reduction: Channel reduction factor (default: 4)
    """
    
    def __init__(self, in_channels: int, reduction: int = 4):
        super(STKE, self).__init__()
        
        mid_channels = in_channels // reduction
        
        # Three parallel convolutional streams with different receptive fields
        # Path a: 1x1 convolution for point-wise features (Eq. 1a)
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Path b: 3x3 convolution for local features (Eq. 1b)
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Path c: Dilated 3x3 convolution for broader context (Eq. 1c)
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2,
                      dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion convolution (Eq. 2)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention unit (Eq. 3)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                           # GAP
            nn.Conv2d(mid_channels, mid_channels // 4, 1),     # W1
            nn.ReLU(inplace=True),                             # δ
            nn.Conv2d(mid_channels // 4, mid_channels, 1),     # W2
            nn.Sigmoid()                                       # σ
        )
        
        # Output projection (We in Eq. 4)
        self.conv_out = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # Residual mapping R(X) - identity or 1x1 mapping
        self.residual = nn.Identity()
        
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
        Forward pass of STKE module.
        
        Args:
            x: Input feature tensor of shape [B, C, H, W]
            
        Returns:
            Enhanced feature tensor of shape [B, C, H, W]
        """
        identity = self.residual(x)  # R(X)
        
        # Multi-scale feature extraction (Eq. 1)
        fa = self.conv_a(x)   # Fine-grained
        fb = self.conv_b(x)   # Local
        fc = self.conv_c(x)   # Broader context
        
        # Concatenate and fuse (Eq. 2)
        f_agg = torch.cat([fa, fb, fc], dim=1)
        f_agg = self.conv_fuse(f_agg)
        
        # Channel attention reweighting (Eq. 3)
        ac = self.channel_attention(f_agg)
        f_weighted = ac * f_agg  # Ac ⊙ Fagg
        
        # Output projection
        f_out = self.conv_out(f_weighted)  # We * (Ac ⊙ Fagg)
        
        # Residual connection (Eq. 4)
        out = identity + f_out  # FSTKE = R(X) + We * (Ac ⊙ Fagg)
        
        return out