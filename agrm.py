"""
Attention-Guided Refinement Module (AGRM)

Reference: Inspired by Adaptive Global Refinement from
IEEE Xplore (2022) - https://ieeexplore.ieee.org/abstract/document/9624979

Implementation follows Equation (19)-(20) of the MAPPR++-Net paper:
  S'2 = (S2 ⊙ CA(S2)) ⊙ SA(S2 ⊙ CA(S2))
  C1 = WConcat(S'2, S2)

Concentrates computational resources on target-specific middle-level regions,
enhancing feature representation at Stage S2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AGRM(nn.Module):
    """
    Attention-Guided Refinement Module.
    
    Applies spatial and channel attention in sequence to adaptively refine
    intermediate representations at Stage S2. This module enhances target
    discriminability and reduces localization ambiguity by progressively
    filtering non-target activations.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (for C1)
        reduction: Channel attention reduction ratio
    """
    
    def __init__(self, in_channels: int, out_channels: int = None,
                 reduction: int = 16):
        super(AGRM, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        
        # Channel Attention (CA)
        mid_ca = max(1, in_channels // reduction)
        self.ca_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_max_pool = nn.AdaptiveMaxPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_ca, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ca, in_channels, 1, bias=False)
        )
        
        # Spatial Attention (SA)
        self.sa_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )
        
        # Weighted Concatenation: C1 = WConcat(S'2, S2) (Eq. 20)
        self.w_concat = nn.Sequential(
            nn.Conv2d(in_channels * 2, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
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
    
    def forward(self, s2: torch.Tensor) -> tuple:
        """
        Forward pass of AGRM module.
        
        Args:
            s2: Stage 2 features [B, C, H, W]
            
        Returns:
            Tuple of (s2_prime, c1):
                s2_prime: Refined features [B, C, H, W]
                c1: Weighted concatenation output [B, out_channels, H, W]
        """
        # Channel Attention (CA)
        ca_avg = self.ca_fc(self.ca_avg_pool(s2))
        ca_max = self.ca_fc(self.ca_max_pool(s2))
        ca_weight = torch.sigmoid(ca_avg + ca_max)
        
        # S2 ⊙ CA(S2)
        s2_ca = s2 * ca_weight
        
        # Spatial Attention (SA)
        sa_avg = torch.mean(s2_ca, dim=1, keepdim=True)
        sa_max, _ = torch.max(s2_ca, dim=1, keepdim=True)
        sa_input = torch.cat([sa_avg, sa_max], dim=1)
        sa_weight = torch.sigmoid(self.sa_conv(sa_input))
        
        # S'2 = (S2 ⊙ CA(S2)) ⊙ SA(S2 ⊙ CA(S2)) (Eq. 19)
        s2_prime = s2_ca * sa_weight
        
        # C1 = WConcat(S'2, S2) (Eq. 20)
        c1 = self.w_concat(torch.cat([s2_prime, s2], dim=1))
        
        return s2_prime, c1