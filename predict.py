"""
Prediction Script for MAPPR++-Net

Usage:
    python predict.py --config configs/mapprpp_irstd1k.yaml \
        --checkpoint checkpoints/best_model.pth \
        --input_dir data/IRSTD-1k/IRSTD1k_Img \
        --output_dir predictions/
"""

import os
import yaml
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import MAPPRPPNet
from utils import visualize_detection, set_seed, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Predict with MAPPR++-Net')
    parser.add_argument('--config', type=str, default='configs/mapprpp_irstd1k.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='predictions/')
    parser.add_argument('--conf_thresh', type=float, default=0.25)
    parser.add_argument('--nms_thresh', type=float, default=0.5)
    return parser.parse_args()


def get_transform(img_size=640):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('SEED', 42))
    device = torch.device(config.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    img_size = config.get('TRAIN', {}).get('IMG_SIZE', 640)
    
    # Model
    model = MAPPRPPNet(config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    
    transform = get_transform(img_size)
    
    # Get image files
    img_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith(('.png', '.jpg', '.bmp', '.tif'))
    ])
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visual'), exist_ok=True)
    
    print(f"\nPredicting on {len(img_files)} images...")
    
    with torch.no_grad():
        for img_file in tqdm(img_files, desc='Predicting'):
            img_path = os.path.join(args.input_dir, img_file)
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Transform
            transformed = transform(image=image_rgb)
            input_tensor = transformed['image'].unsqueeze(0).to(device)
            
            # Predict
            outputs = model(input_tensor)
            detections = model.get_detections(
                outputs,
                conf_thresh=args.conf_thresh,
                nms_thresh=args.nms_thresh,
                img_size=img_size,
            )
            
            # Visualize
            det = detections[0]
            vis_image = visualize_detection(
                image_rgb,
                det['boxes'].numpy(),
                det['scores'].numpy(),
                gt_boxes=None,
                conf_thresh=args.conf_thresh,
            )
            
            # Save
            save_name = os.path.splitext(img_file)[0] + '_pred.png'
            save_path = os.path.join(args.output_dir, 'visual', save_name)
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, vis_bgr)
            
            # Save text results
            txt_path = os.path.join(args.output_dir, os.path.splitext(img_file)[0] + '.txt')
            with open(txt_path, 'w') as f:
                for score, label, box in zip(det['scores'], det['labels'], det['boxes']):
                    if score >= args.conf_thresh:
                        x1, y1, x2, y2 = box.numpy()
                        f.write(f"{int(label)} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
    
    print(f"\nPredictions saved to {args.output_dir}")


if __name__ == '__main__':
    main()