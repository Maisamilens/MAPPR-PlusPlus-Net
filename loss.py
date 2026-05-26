"""
Loss Functions for MAPPR++-Net

Implementation follows Equations (22)-(25) of the MAPPR++-Net paper:
  Ltotal = Ldet + α * Laux + β * Ldn
  Lbbox = λ1 * L1 + λ2 * LGIoU
  where λ1=5.0, λ2=2.0, α=β=1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def box_cxcywh_to_xyxy(x):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Compute Generalized IoU between two sets of boxes.
    
    Args:
        boxes1: [N, 4] in xyxy format
        boxes2: [M, 4] in xyxy format
        
    Returns:
        Giou matrix [N, M]
    """
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # Compute union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    
    iou = inter / union.clamp(min=1e-6)
    
    # Compute enclosing box
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]
    
    giou = iou - (area_enc - union) / area_enc.clamp(min=1e-6)
    
    return giou


class HungarianMatcher(nn.Module):
    """
    Optimal assignment between predictions and ground truth
    using the Hungarian algorithm.
    """
    
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        """
        Args:
            pred_logits: [B, num_queries, num_classes+1]
            pred_boxes: [B, num_queries, 4] in cxcywh format
            targets: List of dicts with 'boxes' and 'labels'
            
        Returns:
            List of (pred_idx, gt_idx) tuples for each image
        """
        from scipy.optimize import linear_sum_assignment
        
        B, Q, _ = pred_logits.shape
        batch_indices = []
        
        for b in range(B):
            # Get valid targets
            gt_boxes = targets[b]['boxes']  # [N, 4]
            gt_labels = targets[b]['labels']  # [N]
            
            if len(gt_labels) == 0:
                batch_indices.append((torch.tensor([], dtype=torch.long),
                                      torch.tensor([], dtype=torch.long)))
                continue
            
            # Classification cost
            out_prob = pred_logits[b].softmax(-1)  # [Q, C+1]
            cost_class = -out_prob[:, gt_labels]  # [Q, N]
            
            # Bbox L1 cost
            out_bbox = pred_boxes[b]  # [Q, 4]
            cost_bbox = torch.cdist(out_bbox, gt_boxes, p=1)  # [Q, N]
            
            # GIoU cost
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(gt_boxes)
            )
            
            # Total cost
            C = (self.cost_class * cost_class +
                 self.cost_bbox * cost_bbox +
                 self.cost_giou * cost_giou)
            
            C = C.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C)
            
            batch_indices.append((
                torch.tensor(row_ind, dtype=torch.long),
                torch.tensor(col_ind, dtype=torch.long)
            ))
        
        return batch_indices


class MAPPRPPLoss(nn.Module):
    """
    Total Loss for MAPPR++-Net.
    
    Ltotal = Ldet + α * Laux + β * Ldn  (Eq. 22)
    Ldet = Lcls + λ1 * L1 + λ2 * LGIoU   (Eqs. 23-25)
    
    Args:
        num_classes: Number of object classes
        matcher: Hungarian matcher
        alpha: Auxiliary loss weight (default: 1.0)
        beta: Denoising loss weight (default: 1.0)
        lambda_l1: L1 loss weight (default: 5.0)
        lambda_giou: GIoU loss weight (default: 2.0)
    """
    
    def __init__(self, num_classes=1, matcher=None,
                 alpha=1.0, beta=1.0,
                 lambda_l1=5.0, lambda_giou=2.0):
        super(MAPPRPPLoss, self).__init__()
        
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher(
            cost_class=1.0, cost_bbox=lambda_l1, cost_giou=lambda_giou
        )
        self.alpha = alpha
        self.beta = beta
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        
        # Classification loss weight (for no-object class)
        self.empty_weight = torch.ones(num_classes + 1)
        self.empty_weight[-1] = 0.1  # Lower weight for background
    
    def forward(self, pred_logits, pred_boxes, targets,
                aux_outputs=None, dn_outputs=None):
        """
        Compute total loss.
        
        Args:
            pred_logits: [B, Q, C+1] classification predictions
            pred_boxes: [B, Q, 4] bbox predictions (cxcywh)
            targets: List of dicts with 'boxes' and 'labels'
            aux_outputs: Auxiliary decoder outputs
            dn_outputs: Denoising training outputs
            
        Returns:
            Dictionary of losses
        """
        device = pred_logits.device
        self.empty_weight = self.empty_weight.to(device)
        
        # Match predictions to targets
        indices = self.matcher(pred_logits, pred_boxes, targets)
        
        # Compute main detection loss
        losses = self._compute_loss(pred_logits, pred_boxes, targets,
                                     indices, prefix='')
        
        # Auxiliary losses (from intermediate decoder layers)
        if aux_outputs is not None:
            for i, aux in enumerate(aux_outputs):
                aux_indices = self.matcher(
                    aux['pred_logits'], aux['pred_boxes'], targets
                )
                aux_losses = self._compute_loss(
                    aux['pred_logits'], aux['pred_boxes'], targets,
                    aux_indices, prefix=f'aux_{i}_'
                )
                for k, v in aux_losses.items():
                    losses[k] = losses.get(k, torch.tensor(0.0, device=device)) + \
                                self.alpha * v  # α * Laux
        
        # Denoising losses
        if dn_outputs is not None:
            for i, dn in enumerate(dn_outputs):
                dn_indices = self.matcher(
                    dn['pred_logits'], dn['pred_boxes'], targets
                )
                dn_losses = self._compute_loss(
                    dn['pred_logits'], dn['pred_boxes'], targets,
                    dn_indices, prefix=f'dn_{i}_'
                )
                for k, v in dn_losses.items():
                    losses[k] = losses.get(k, torch.tensor(0.0, device=device)) + \
                                self.beta * v  # β * Ldn
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _compute_loss(self, pred_logits, pred_boxes, targets, indices, prefix=''):
        """Compute detection loss components."""
        device = pred_logits.device
        
        # Classification loss (cross-entropy)
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,  # background class
            dtype=torch.int64,
            device=device
        )
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) > 0:
                target_classes[b, pred_idx] = targets[b]['labels'][gt_idx]
        
        loss_cls = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight
        )
        
        # Bounding box losses
        loss_bbox = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
        num_boxes = 0
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) == 0:
                continue
            
            pred_b = pred_boxes[b, pred_idx]  # [N, 4]
            gt_b = targets[b]['boxes'][gt_idx]  # [N, 4]
            
            # L1 loss (Eq. 24)
            loss_bbox += F.l1_loss(pred_b, gt_b, reduction='sum')
            
            # GIoU loss (Eq. 25)
            pred_xyxy = box_cxcywh_to_xyxy(pred_b)
            gt_xyxy = box_cxcywh_to_xyxy(gt_b)
            giou = generalized_box_iou(pred_xyxy, gt_xyxy)
            loss_giou += (1 - torch.diag(giou)).sum()
            
            num_boxes += len(gt_idx)
        
        num_boxes = max(num_boxes, 1)
        
        losses = {
            f'{prefix}loss_cls': loss_cls,
            f'{prefix}loss_bbox': self.lambda_l1 * loss_bbox / num_boxes,
            f'{prefix}loss_giou': self.lambda_giou * loss_giou / num_boxes,
        }
        
        return losses