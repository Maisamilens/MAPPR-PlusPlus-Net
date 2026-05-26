"""Miscellaneous Utilities for MAPPR++-Net"""

import os
import torch
import random
import numpy as np
from typing import Dict


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: Dict, filepath: str, is_best: bool = False):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), 'best_model.pth')
        torch.save(state, best_path)


def load_checkpoint(filepath: str, model, optimizer=None):
    """Load model checkpoint."""
    if not os.path.exists(filepath):
        print(f"[Checkpoint] No checkpoint found at {filepath}")
        return {}
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"[Checkpoint] Loaded checkpoint from {filepath}")
    return checkpoint


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count