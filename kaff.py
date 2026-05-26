"""
Key-Aware Feature Fusion (KAFF) Module

Reference: Inspired by key-aware feature fusion concepts from
IEEE Xplore (2019) - https://ieeexplore.ieee.org/abstract/document/8732991

Implementation follows Equations (5)-(8) of the MAPPR++-Net paper:
  Fconcat = Concat(X, FSTKE)
  G = σ(Conv3×3(Fconcat))
  Ftr = δ(Conv3×3(Fconcat))
  FKAFF = (1-G) ⊙ FSTKE + G ⊙ Ftr

The module employs dual-branch gated fusion between the original and
STKE-enhanced features, adaptively weighting each branch to preserve
salient target information and suppress background noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KAFF(nn.Module):
    """
    Key-Aware Feature Fusion Module.
    
    Adaptively merges original shallow features with STKE-enhanced
    representations using a dual-branch gated fusion mechanism.
    
    Args:
        in_channels: Number of input feature channels
    """
    
    def __init__(self, in_channels: int):
        super(KAFF, self).__init__()
        
        # Concatenated features have 2x channels (original + STKE enhanced)
        concat_channels = in_channels * 2
        
        # Gating branch: G = σ(Conv3×3(Fconcat))  (Eq. 6)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(concat_channels, in_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        # Transform branch: Ftr = δ(Conv3×3(Fconcat))  (Eq. 7)
        self.transform_conv = nn.Sequential(
            nn.Conv2d(concat_channels, in_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
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
    
    def forward(self, x: torch.Tensor, f_stke: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of KAFF module.
        
        Args:
            x: Original shallow features [B, C, H, W]
            f_stke: STKE-enhanced features [B, C, H, W]
            
        Returns:
            Fused feature tensor [B, C, H, W]
        """
        # Concatenate original and STKE features (Eq. 5)
        f_concat = torch.cat([x, f_stke], dim=1)
        
        # Compute gate weights (Eq. 6)
        g = self.gate_conv(f_concat)  # σ(Conv3×3(Fconcat))
        
        # Compute transformed features (Eq. 7)
        f_tr = self.transform_conv(f_concat)  # δ(Conv3×3(Fconcat))
        
        # Gated fusion (Eq. 8)
        out = (1 - g) * f_stke + g * f_tr  # FKAFF = (1-G)⊙FSTKE + G⊙Ftr
        
        return out