"""
Evaluation Metrics for MAPPR++-Net

Implements all metrics from the paper:
- Precision, Recall, F1 (Eqs. 26-28)
- mAP50, mAP50-95 (Eqs. 29-31)
- Pd (Probability of Detection, Eq. 32)
- Fa (False Alarm Rate, Eq. 33)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes in xyxy format."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    
    return inter / np.maximum(union, 1e-6)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision from recall-precision curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_metrics(
    all_detections: List[Dict],
    all_targets: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 1,
) -> Dict:
    """
    Compute detection metrics.
    
    Args:
        all_detections: List of detection dicts per image
        all_targets: List of target dicts per image
        iou_threshold: IoU threshold for TP/FP determination
        num_classes: Number of object classes
        
    Returns:
        Dictionary with Precision, Recall, F1, mAP50, mAP50-95
    """
    results = defaultdict(list)
    
    for iou_thresh_mult in [0.5]:  # mAP50
        tp_list, fp_list, fn_list = [], [], []
        ap_per_class = []
        
        for cls_id in range(num_classes):
            all_scores = []
            all_tp = []
            all_fp = []
            n_gt = 0
            
            for det, tgt in zip(all_detections, all_targets):
                # Get detections for this class
                if len(det['scores']) > 0:
                    mask = det['labels'] == cls_id
                    det_scores = det['scores'][mask]
                    det_boxes = det['boxes'][mask]
                else:
                    det_scores = np.array([])
                    det_boxes = np.array([]).reshape(0, 4)
                
                # Get targets for this class
                if len(tgt['labels']) > 0:
                    tgt_mask = tgt['labels'] == cls_id
                    tgt_boxes = tgt['boxes'][tgt_mask]
                else:
                    tgt_boxes = np.array([]).reshape(0, 4)
                
                n_gt += len(tgt_boxes)
                
                if len(det_boxes) == 0:
                    continue
                
                # Sort by confidence
                sort_idx = np.argsort(-det_scores)
                det_scores = det_scores[sort_idx]
                det_boxes = det_boxes[sort_idx]
                
                if len(tgt_boxes) > 0:
                    iou_mat = compute_iou_matrix(det_boxes, tgt_boxes)
                else:
                    iou_mat = np.zeros((len(det_boxes), 0))
                
                matched_gt = set()
                
                for d_idx in range(len(det_boxes)):
                    all_scores.append(det_scores[d_idx])
                    
                    if len(tgt_boxes) > 0:
                        best_iou = np.max(iou_mat[d_idx])
                        best_gt = np.argmax(iou_mat[d_idx])
                        
                        if best_iou >= iou_thresh_mult and best_gt not in matched_gt:
                            all_tp.append(1)
                            all_fp.append(0)
                            matched_gt.add(best_gt)
                        else:
                            all_tp.append(0)
                            all_fp.append(1)
                    else:
                        all_tp.append(0)
                        all_fp.append(1)
            
            if n_gt > 0 and len(all_scores) > 0:
                all_scores = np.array(all_scores)
                all_tp = np.array(all_tp)
                all_fp = np.array(all_fp)
                
                sort_idx = np.argsort(-all_scores)
                all_tp = all_tp[sort_idx]
                all_fp = all_fp[sort_idx]
                
                tp_cumsum = np.cumsum(all_tp)
                fp_cumsum = np.cumsum(all_fp)
                
                recall = tp_cumsum / n_gt
                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                
                ap = compute_ap(recall, precision)
                ap_per_class.append(ap)
            elif n_gt > 0:
                ap_per_class.append(0.0)
        
        if len(ap_per_class) > 0:
            results['mAP50'].append(np.mean(ap_per_class))
    
    # Compute mAP50-95
    ap_per_thresh = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        ap_per_class = []
        
        for cls_id in range(num_classes):
            all_scores = []
            all_tp = []
            all_fp =