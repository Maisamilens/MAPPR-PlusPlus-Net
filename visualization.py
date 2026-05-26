"""
Visualization Utilities for MAPPR++-Net

Generates all paper figures:
- Detection visualizations (Fig. 9, 11)
- 3D confidence bar charts (Fig. 10)
- Training curves (Fig. 12)
- Radar charts (Fig. 8)
- Ablation study bar charts (Table V)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional

matplotlib.use('Agg')


def visualize_detection(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    gt_boxes: np.ndarray = None,
    save_path: str = None,
    conf_thresh: float = 0.3,
    target_color: tuple = (0, 255, 0),
    gt_color: tuple = (0, 0, 255),
    miss_color: tuple = (255, 0, 0),
    false_alarm_color: tuple = (0, 255, 255),
):
    """
    Visualize detection results with colored circles (as in Fig. 9, 11).
    
    Colors:
    - Green: Correctly detected targets
    - Blue: Missed detections
    - Yellow: False alarms
    - Red circle: Ground truth
    
    Args:
        image: Input image (H, W, 3) in RGB format
        boxes: Predicted boxes [N, 4] in xyxy format
        scores: Confidence scores [N]
        gt_boxes: Ground truth boxes [M, 4] in xyxy format
        save_path: Path to save visualization
        conf_thresh: Confidence threshold
    """
    img_vis = image.copy()
    if img_vis.max() <= 1.0:
        img_vis = (img_vis * 255).astype(np.uint8)
    
    # Draw ground truth
    if gt_boxes is not None:
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = map(int, gt_box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max((x2 - x1), (y2 - y1)) // 2 + 5
            cv2.circle(img_vis, (cx, cy), radius, gt_color, 2)
    
    # Draw predictions
    if boxes is not None and len(boxes) > 0:
        for box, score in zip(boxes, scores):
            if score < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max((x2 - x1), (y2 - y1)) // 2 + 5
            
            # Determine if TP or FP
            is_tp = False
            if gt_boxes is not None and len(gt_boxes) > 0:
                for gt_box in gt_boxes:
                    ix1 = max(x1, int(gt_box[0]))
                    iy1 = max(y1, int(gt_box[1]))
                    ix2 = min(x2, int(gt_box[2]))
                    iy2 = min(y2, int(gt_box[3]))
                    
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_pred = max(0, x2 - x1) * max(0, y2 - y1)
                    area_gt = max(0, int(gt_box[2]) - int(gt_box[0])) * \
                              max(0, int(gt_box[3]) - int(gt_box[1]))
                    union = area_pred + area_gt - inter
                    iou = inter / max(union, 1)
                    
                    if iou > 0.3:
                        is_tp = True
                        break
            
            color = target_color if is_tp else false_alarm_color
            cv2.circle(img_vis, (cx, cy), radius, color, 2)
            
            # Add confidence score
            cv2.putText(img_vis, f"{score:.2f}", (x2 + 3, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw missed detections (blue circles for unmatched GT)
    if gt_boxes is not None and boxes is not None and len(boxes) > 0:
        for gt_box in gt_boxes:
            gx1, gy1, gx2, gy2 = map(int, gt_box)
            gcx, gcy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
            matched = False
            
            for box, score in zip(boxes, scores):
                if score < conf_thresh:
                    continue
                px1, py1, px2, py2 = map(int, box)
                ix1 = max(px1, gx1)
                iy1 = max(py1, gy1)
                ix2 = min(px2, gx2)
                iy2 = min(py2, gy2)
                
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_p = max(0, px2 - px1) * max(0, py2 - py1)
                area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
                union = area_p + area_g - inter
                iou = inter / max(union, 1)
                
                if iou > 0.3:
                    matched = True
                    break
            
            if not matched:
                radius = max((gx2 - gx1), (gy2 - gy1)) // 2 + 5
                cv2.circle(img_vis, (gcx, gcy), radius, miss_color, 2)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert RGB to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
    
    return img_vis


def visualize_3d_confidence(
    methods: List[str],
    target_ids: List[str],
    confidence_scores: np.ndarray,
    save_path: str = None,
):
    """
    3D bar chart comparison of confidence scores (Fig. 10).
    
    Args:
        methods: List of method names
        target_ids: List of target identifiers
        confidence_scores: 2D array [num_methods, num_targets]
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(len(target_ids))
    y = np.arange(len(methods))
    X, Y = np.meshgrid(x, y)
    
    xpos = X.ravel()
    ypos = Y.ravel()
    zpos = np.zeros_like(xpos)
    
    dx = dy = 0.4
    dz = confidence_scores.ravel()
    
    colors = plt.cm.viridis(dz / max(dz.max(), 1))
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8)
    
    ax.set_xlabel('Targets')
    ax.set_ylabel('Methods')
    ax.set_zlabel('Confidence')
    ax.set_xticks(x)
    ax.set_xticklabels(target_ids, rotation=45)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_title('3D Confidence Score Comparison')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    return fig


def visualize_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: str = None,
):
    """
    Training curves for P, R, mAP50, mAP50-95 (Fig. 12).
    
    Args:
        metrics_history: Dictionary with metric names as keys and lists of values
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    plot_configs = [
        ('Precision', axes[0, 0]),
        ('Recall', axes[0, 1]),
        ('mAP50', axes[1, 0]),
        ('mAP50-95', axes[1, 1]),
    ]
    
    for metric_name, ax in plot_configs:
        if metric_name in metrics_history:
            values = metrics_history[metric_name]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, 'b-', linewidth=2, label=metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Epoch')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    return fig


def visualize_radar_chart(
    methods: List[str],
    metrics: List[str],
    values: np.ndarray,
    save_path: str = None,
):
    """
    Radar chart for multi-dataset comparison (Fig. 8).
    
    Args:
        methods: List of method names
        metrics: List of metric names
        values: 2D array [num_methods, num_metrics]
        save_path: Path to save figure
    """
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for i, (method, method_values) in enumerate(zip(methods, values)):
        vals = method_values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=method, color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Performance Comparison')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    return fig


def visualize_ablation(
    method_names: List[str],
    metrics: Dict[str, List[float]],
    save_path: str = None,
):
    """
    Ablation study bar charts (Table V visualization).
    
    Args:
        method_names: List of ablation configurations
        metrics: Dictionary with metric names as keys and lists of values
        save_path: Path to save figure
    """
    x = np.arange(len(method_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_names = list(metrics.keys())
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
    
    for i, metric_name in enumerate(metric_names):
        offset = width * (i - len(metric_names) / 2 + 0.5)
        bars = ax.bar(x + offset, metrics[metric_name], width,
                      label=metric_name, color=colors[i % len(colors)])
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score (%)')
    ax.set_title('Ablation Study Results')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    return fig