"""
Training Script for MAPPR++-Net

Usage:
    python train.py --config configs/mapprpp_irstd1k.yaml
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models import MAPPRPPNet
from datasets import get_dataloaders
from utils import set_seed, save_checkpoint, AverageMeter, compute_metrics
from utils.misc import count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description='Train MAPPR++-Net')
    parser.add_argument('--config', type=str, default='configs/mapprpp_irstd1k.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_optimizer(model, config):
    train_cfg = config['TRAIN']
    
    param_dicts = [
        {
            'params': [p for n, p in model.named_parameters()
                       if 'marfpp' not in n and p.requires_grad],
            'lr': train_cfg['LR'],
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if 'marfpp' in n and p.requires_grad],
            'lr': train_cfg['LR'] * 0.1,  # Lower LR for MARF++ modules
        },
    ]
    
    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=train_cfg['LR'],
        weight_decay=train_cfg['WEIGHT_DECAY'],
    )
    
    return optimizer


def build_scheduler(optimizer, config):
    train_cfg = config['TRAIN']
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['EPOCHS'],
        eta_min=train_cfg.get('WARMUP_LR', 1e-6),
    )
    
    return scheduler


def train_one_epoch(model, dataloader, optimizer, device, epoch, writer, config):
    model.train()
    loss_meter = AverageMeter('Loss')
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for i, (images, targets) in enumerate(pbar):
        images = images.to(device)
        
        # Move targets to device
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)
        
        # Forward pass
        outputs = model(images, targets)
        
        # Get losses
        losses = outputs.get('losses', {})
        total_loss = losses.get('total', torch.tensor(0.0, device=device))
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        optimizer.step()
        
        # Update meters
        loss_meter.update(total_loss.item(), images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'cls': f'{losses.get("loss_cls", torch.tensor(0)).item():.4f}',
            'bbox': f'{losses.get("loss_bbox", torch.tensor(0)).item():.4f}',
            'giou': f'{losses.get("loss_giou", torch.tensor(0)).item():.4f}',
        })
        
        # TensorBoard logging
        global_step = epoch * len(dataloader) + i
        if i % 50 == 0:
            writer.add_scalar('Train/TotalLoss', total_loss.item(), global_step)
            for loss_name, loss_val in losses.items():
                if isinstance(loss_val, torch.Tensor):
                    writer.add_scalar(f'Train/{loss_name}', loss_val.item(), global_step)
    
    return loss_meter.avg


@torch.no_grad()
def validate(model, dataloader, device, epoch, writer, config):
    model.eval()
    
    all_detections = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    for images, targets in pbar:
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Get detections
        detections = model.get_detections(
            outputs,
            conf_thresh=config['TEST']['CONF_THRESH'],
            nms_thresh=config['TEST']['NMS_THRESH'],
            img_size=images.shape[-1],
        )
        
        # Prepare for metric computation
        for det, tgt in zip(detections, targets):
            det_np = {
                'scores': det['scores'].numpy(),
                'labels': det['labels'].numpy(),
                'boxes': det['boxes'].numpy(),
            }
            tgt_np = {
                'labels': tgt['labels'].numpy(),
                'boxes': tgt['boxes'].numpy() * images.shape[-1],  # Scale up
            }
            all_detections.append(det_np)
            all_targets.append(tgt_np)
    
    # Compute metrics
    metrics = compute_metrics(
        all_detections, all_targets,
        iou_threshold=0.5,
        num_classes=config['DATASET']['NUM_CLASSES'],
    )
    
    # TensorBoard logging
    for metric_name, metric_val in metrics.items():
        writer.add_scalar(f'Val/{metric_name}', metric_val, epoch)
    
    print(f"\n[Validation] Epoch {epoch}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
    
    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Setup
    set_seed(config.get('SEED', 42))
    device = torch.device(config.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Create directories
    os.makedirs(config['PATHS']['CHECKPOINT_DIR'], exist_ok=True)
    os.makedirs(config['PATHS']['LOG_DIR'], exist_ok=True)
    
    # Data
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Model
    model = MAPPRPPNet(config).to(device)
    print(f"\nModel Parameters: {count_parameters(model):,}")
    
    # Optimizer and Scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=config['PATHS']['LOG_DIR'])
    
    # Training loop
    best_map = 0.0
    epochs = config['TRAIN']['EPOCHS']
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                      epoch, writer, config)
        
        # Validate
        if epoch % 5 == 0 or epoch == epochs:
            metrics = validate(model, val_loader, device, epoch, writer, config)
            
            # Save best model
            current_map = metrics.get('mAP50', 0)
            is_best = current_map > best_map
            if is_best:
                best_map = current_map
            
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_map': best_map,
                    'config': config,
                },
                os.path.join(config['PATHS']['CHECKPOINT_DIR'], f'checkpoint_epoch_{epoch}.pth'),
                is_best=is_best,
            )
        
        # Update scheduler
        scheduler.step()
        
        # Log learning rate
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
    
    writer.close()
    print(f"\nTraining complete! Best mAP50: {best_map:.2f}")


if __name__ == '__main__':
    main()