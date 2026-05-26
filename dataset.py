"""
IRSTD-1k Dataset Handler for MAPPR++-Net

Handles:
- Loading IR images and mask labels
- Converting segmentation masks to YOLO bounding box format
- Train/Val/Test splitting (6:2:2 ratio)
- Data augmentation with Albumentations
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse


class IRSTD1kDataset(Dataset):
    """
    IRSTD-1k Dataset for Infrared Small Target Detection.
    
    Args:
        img_dir: Path to IRSTD1k_Img directory
        mask_dir: Path to IRSTD1k_Label directory
        label_dir: Path to YOLO-format label directory (generated)
        split_file: Path to split file (trainval.txt or test.txt)
        split: One of 'train', 'val', 'test'
        train_ratio: Ratio of training data from trainval split
        transforms: Albumentations transforms
        img_size: Input image size
    """
    
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        label_dir: str,
        split_file: str = None,
        split: str = 'train',
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        transforms=None,
        img_size: int = 640,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.split = split
        self.img_size = img_size
        self.transforms = transforms
        self.num_classes = 1
        
        # Load image IDs from split file
        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                all_ids = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Get all image IDs from img_dir
            all_ids = [
                os.path.splitext(f)[0]
                for f in os.listdir(img_dir)
                if f.endswith(('.png', '.jpg', '.bmp'))
            ]
            all_ids.sort()
        
        # Split data: 60% train, 20% val, 20% test
        n_total = len(all_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if split == 'train':
            self.image_ids = all_ids[:n_train]
        elif split == 'val':
            self.image_ids = all_ids[n_train:n_train + n_val]
        elif split == 'test':
            self.image_ids = all_ids[n_train + n_val:]
        else:
            self.image_ids = all_ids
        
        print(f"[IRSTD-1k] {split} split: {len(self.image_ids)} images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            img_path = os.path.join(self.img_dir, f"{img_id}.bmp")
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert grayscale to 3-channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Load mask for segmentation metrics
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask_path = os.path.join(self.mask_dir, f"{img_id}.jpg")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        h_orig, w_orig = image.shape[:2]
        
        # Extract bounding boxes from mask
        boxes, labels = self._mask_to_boxes(mask)
        
        # Apply transforms
        if self.transforms:
            if len(boxes) > 0:
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes,
                    labels=labels
                )
                image = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['labels']
            else:
                transformed = self.transforms(image=image)
                image = transformed['image']
                boxes = []
                labels = []
        
        # Normalize boxes to [0, 1] range (cx, cy, w, h format)
        if len(boxes) > 0:
            boxes_tensor = torch.zeros((len(boxes), 4), dtype=torch.float32)
            for i, bbox in enumerate(boxes):
                if isinstance(bbox, (list, tuple)):
                    x1, y1, x2, y2 = bbox[:4]
                else:
                    x1, y1, x2, y2 = bbox
                # Convert to cx, cy, w, h normalized
                cx = ((x1 + x2) / 2.0) / self.img_size
                cy = ((y1 + y2) / 2.0) / self.img_size
                w = (x2 - x1) / self.img_size
                h = (y2 - y1) / self.img_size
                # Clamp to [0, 1]
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0.001, min(1, w))
                h = max(0.001, min(1, h))
                boxes_tensor[i] = torch.tensor([cx, cy, w, h])
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        
        # Resize mask for segmentation evaluation
        if mask is not None:
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = (mask > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, self.img_size, self.img_size))
        
        target = {
            'boxes': boxes_tensor,        # [N, 4] cx, cy, w, h normalized
            'labels': labels_tensor,       # [N]
            'masks': mask_tensor,          # [1, H, W]
            'image_id': torch.tensor(idx),
            'orig_size': torch.tensor([h_orig, w_orig]),
            'img_id_str': img_id,
        }
        
        return image, target
    
    def _mask_to_boxes(self, mask: np.ndarray) -> Tuple[List, List]:
        """Convert segmentation mask to bounding boxes."""
        if mask is None:
            return [], []
        
        binary = (mask > 127).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        boxes = []
        labels_list = []
        
        for i in range(1, num_labels):  # Skip background (0)
            x, y, w, h, area = stats[i]
            if area < 2:  # Skip tiny artifacts
                continue
            # Add small padding
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(mask.shape[1], x + w + pad)
            y2 = min(mask.shape[0], y + h + pad)
            
            # Convert to resized coordinates
            scale_x = self.img_size / mask.shape[1]
            scale_y = self.img_size / mask.shape[0]
            boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
            labels_list.append(1)  # Class 1: target
        
        return boxes, labels_list


def get_train_transforms(img_size: int = 640):
    """Training augmentations."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.1
        )
    )


def get_val_transforms(img_size: int = 640):
    """Validation/Test augmentations."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.0
        )
    )


def collate_fn(batch):
    """Custom collate function for variable-size targets."""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, 0)
    
    return images, targets


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    img_size = config.get('TRAIN', {}).get('IMG_SIZE', 640)
    
    train_dataset = IRSTD1kDataset(
        img_dir=config['DATASET']['IMG_DIR'],
        mask_dir=config['DATASET']['MASK_DIR'],
        label_dir=config['DATASET']['LABEL_DIR'],
        split_file=config['DATASET']['TRAIN_FILE'],
        split='train',
        train_ratio=config['DATASET']['TRAIN_RATIO'],
        val_ratio=config['DATASET']['VAL_RATIO'],
        transforms=get_train_transforms(img_size),
        img_size=img_size,
    )
    
    val_dataset = IRSTD1kDataset(
        img_dir=config['DATASET']['IMG_DIR'],
        mask_dir=config['DATASET']['MASK_DIR'],
        label_dir=config['DATASET']['LABEL_DIR'],
        split_file=config['DATASET']['TRAIN_FILE'],
        split='val',
        train_ratio=config['DATASET']['TRAIN_RATIO'],
        val_ratio=config['DATASET']['VAL_RATIO'],
        transforms=get_val_transforms(img_size),
        img_size=img_size,
    )
    
    test_dataset = IRSTD1kDataset(
        img_dir=config['DATASET']['IMG_DIR'],
        mask_dir=config['DATASET']['MASK_DIR'],
        label_dir=config['DATASET']['LABEL_DIR'],
        split_file=config['DATASET']['TEST_FILE'],
        split='test',
        train_ratio=config['DATASET']['TRAIN_RATIO'],
        val_ratio=config['DATASET']['VAL_RATIO'],
        transforms=get_val_transforms(img_size),
        img_size=img_size,
    )
    
    batch_size = config['TRAIN']['BATCH_SIZE']
    num_workers = config['TRAIN']['NUM_WORKERS']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def prepare_labels(img_dir: str, mask_dir: str, output_dir: str):
    """
    Convert segmentation masks to YOLO-format bounding box labels.
    Saves one .txt file per image with bounding box annotations.
    Format: class_id cx cy w h (normalized 0-1)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith(('.png', '.jpg', '.bmp'))
    ])
    
    for img_file in img_files:
        img_id = os.path.splitext(img_file)[0]
        mask_path = os.path.join(mask_dir, f"{img_id}.png")
        
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, f"{img_id}.jpg")
        
        if not os.path.exists(mask_path):
            continue
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        h, w = mask.shape
        binary = (mask > 127).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        lines = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < 2:
                continue
            # YOLO format: class cx cy w h (normalized)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        
        label_path = os.path.join(output_dir, f"{img_id}.txt")
        with open(label_path, 'w') as f:
            f.writelines(lines)
    
    print(f"[Label Preparation] Generated labels in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare IRSTD-1k labels')
    parser.add_argument('--prepare_labels', action='store_true')
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    if args.prepare_labels:
        prepare_labels(args.img_dir, args.mask_dir, args.output_dir)