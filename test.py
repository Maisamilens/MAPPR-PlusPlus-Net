"""
Testing Script for MAPPR++-Net

Usage:
    python test.py --config configs/mapprpp_irstd1k.yaml --checkpoint checkpoints/best_model.pth
"""

import os
import yaml
import argparse
import torch
import json
from tqdm import tqdm
import numpy as np

from models import MAPPRPPNet
from datasets import get_dataloaders
from utils import compute_metrics, compute_pd_fa, set_seed, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Test MAPPR++-Net')
    parser.add_argument('--config', type=str, default='configs/mapprpp_irstd1k.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    parser.add_argument('--save_results', action='store_true', help='Save per-image results')
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('SEED', 42))
    device = torch.device(config.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Data
    _, val_loader, test_loader = get_dataloaders(config)
    dataloader = test_loader if args.split == 'test' else val_loader
    
    # Model
    model = MAPPRPPNet(config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    
    print(f"\nTesting on {args.split} set ({len(dataloader.dataset)} images)...")
    
    all_detections = []
    all_targets = []
    all_image_ids = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Testing'):
            images = images.to(device)
            
            outputs = model(images)
            detections = model.get_detections(
                outputs,
                conf_thresh=config['TEST']['CONF_THRESH'],
                nms_thresh=config['TEST']['NMS_THRESH'],
                img_size=images.shape[-1],
            )
            
            for det, tgt in zip(detections, targets):
                det_np = {
                    'scores': det['scores'].numpy(),
                    'labels': det['labels'].numpy(),
                    'boxes': det['boxes'].numpy(),
                }
                tgt_np = {
                    'labels': tgt['labels'].numpy(),
                    'boxes': tgt['boxes'].numpy() * images.shape[-1],
                }
                all_detections.append(det_np)
                all_targets.append(tgt_np)
                all_image_ids.append(tgt.get('img_id_str', ''))
    
    # Compute metrics
    metrics = compute_metrics(
        all_detections, all_targets,
        iou_threshold=0.5,
        num_classes=config['DATASET']['NUM_CLASSES'],
    )
    
    pd_fa = compute_pd_fa(
        all_detections, all_targets,
        img_size=config.get('TRAIN', {}).get('IMG_SIZE', 640),
    )
    
    # Print results
    print("\n" + "="*50)
    print(f"  Test Results ({args.split} set)")
    print("="*50)
    print(f"  Precision:   {metrics['Precision']:.1f}%")
    print(f"  Recall:      {metrics['Recall']:.1f}%")
    print(f"  F1:          {metrics['F1']:.1f}%")
    print(f"  mAP50:       {metrics['mAP50']*100:.1f}%")
    print(f"  mAP50-95:    {metrics['mAP50-95']*100:.1f}%")
    print(f"  Pd:          {pd_fa['Pd']:.1f}%")
    print(f"  Fa:          {pd_fa['Fa']:.4f}")
    print("="*50)
    
    # Save results
    if args.save_results:
        results = {
            'split': args.split,
            'checkpoint': args.checkpoint,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'pd_fa': {k: float(v) for k, v in pd_fa.items()},
            'per_image': [],
        }
        
        for img_id, det, tgt in zip(all_image_ids, all_detections, all_targets):
            results['per_image'].append({
                'image_id': img_id if isinstance(img_id, str) else str(img_id),
                'num_detections': int(len(det['scores'])),
                'num_gt': int(len(tgt['labels'])),
                'scores': det['scores'].tolist(),
            })
        
        save_path = os.path.join(config['PATHS']['CHECKPOINT_DIR'], 'test_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()