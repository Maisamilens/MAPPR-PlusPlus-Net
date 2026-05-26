"""
ResNet18 Backbone with MARF++ Block Integration

The backbone replaces standard BasicBlock with MARF++ blocks at
Stages S1-S4 for enhanced multi-path attention feature extraction.
Based on the RT-DETR framework (Zhao et al., CVPR 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .marfpp import MARFPlusPlus


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class MARFPPBlock(nn.Module):
    """
    ResNet-style block that uses MARF++ instead of standard BasicBlock.
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 use_marfpp=True, use_dps=True, use_cpi=True):
        super(MARFPPBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
        # MARF++ attention
        self.use_marfpp = use_marfpp
        if use_marfpp:
            self.marfpp = MARFPlusPlus(
                out_channels,
                num_paths=3,
                use_dps=use_dps,
                use_cpi=use_cpi,
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        if self.use_marfpp:
            out = self.marfpp(out)
        
        return out


class ResNet18MARFPP(nn.Module):
    """
    ResNet18 Backbone with MARF++ blocks at each stage.
    
    Returns multi-scale features from S1-S4 for the detection framework.
    
    Args:
        pretrained: Whether to load ImageNet pretrained weights
        use_marfpp: Whether to use MARF++ blocks
        use_dps: Enable Dynamic Path Selection in MARF++
        use_cpi: Enable Cross-Path Interaction in MARF++
    """
    
    def __init__(self, pretrained=True, use_marfpp=True,
                 use_dps=True, use_cpi=True):
        super(ResNet18MARFPP, self).__init__()
        
        self.in_channels = 64
        self.use_marfpp = use_marfpp
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage layers with optional MARF++
        BlockClass = MARFPPBlock if use_marfpp else BasicBlock
        
        self.layer1 = self._make_layer(BlockClass, 64, 2, stride=1,
                                        use_marfpp=use_marfpp,
                                        use_dps=use_dps, use_cpi=use_cpi)
        self.layer2 = self._make_layer(BlockClass, 128, 2, stride=2,
                                        use_marfpp=use_marfpp,
                                        use_dps=use_dps, use_cpi=use_cpi)
        self.layer3 = self._make_layer(BlockClass, 256, 2, stride=2,
                                        use_marfpp=use_marfpp,
                                        use_dps=use_dps, use_cpi=use_cpi)
        self.layer4 = self._make_layer(BlockClass, 512, 2, stride=2,
                                        use_marfpp=use_marfpp,
                                        use_dps=use_dps, use_cpi=use_cpi)
        
        # Load pretrained weights if available
        if pretrained and not use_marfpp:
            self._load_pretrained()
        elif pretrained and use_marfpp:
            self._load_pretrained_partial()
        
        self._init_remaining_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride,
                     use_marfpp=True, use_dps=True, use_cpi=True):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride,
                            downsample, use_marfpp=use_marfpp,
                            use_dps=use_dps, use_cpi=use_cpi))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels,
                                use_marfpp=use_marfpp,
                                use_dps=use_dps, use_cpi=use_cpi))
        
        return nn.Sequential(*layers)
    
    def _load_pretrained_partial(self):
        """Load pretrained ResNet18 weights for non-MARF++ parts."""
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            
            # Load compatible weights
            pretrained_dict = model.state_dict()
            model_dict = self.state_dict()
            
            # Filter out MARF++ specific keys
            compatible_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_dict[k] = v
            
            model_dict.update(compatible_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"[Backbone] Loaded {len(compatible_dict)}/{len(model_dict)} pretrained weights")
        except Exception as e:
            print(f"[Backbone] Could not load pretrained weights: {e}")
    
    def _load_pretrained(self):
        """Load full pretrained ResNet18 weights."""
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.load_state_dict(model.state_dict(), strict=False)
        except Exception as e:
            print(f"[Backbone] Could not load pretrained weights: {e}")
    
    def _init_remaining_weights(self):
        """Initialize any remaining uninit weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None and m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Tuple of (S1, S2, S3, S4) feature maps
        """
        # Initial
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Stage outputs
        s1 = self.layer1(x)    # [B, 64, H/4, W/4]
        s2 = self.layer2(s1)   # [B, 128, H/8, W/8]
        s3 = self.layer3(s2)   # [B, 256, H/16, W/16]
        s4 = self.layer4(s3)   # [B, 512, H/32, W/32]
        
        return s1, s2, s3, s4