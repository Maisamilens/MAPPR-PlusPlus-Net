"""
MAPPR++-Net: Complete Network Architecture

Multi-Path Attention Residual and Perception Progressive Refinement
Plus Plus Network for Infrared Small Target Detection.

Combines all modules:
- ResNet18 + MARF++ Backbone
- STKE + KAFF (Shallow Feature Knowledge Complement - SKC)
- CFPR (Cross-Feature Progressive Refinement)
- AGRM (Attention-Guided Refinement Module at S2)
- CARM (Context-Aware Refinement Module at S3)
- RT-DETR Hybrid Encoder + Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .backbone import ResNet18MARFPP
from .stke import STKE
from .kaff import KAFF
from .cfpr import CFPR
from .agrm import AGRM
from .carm import CARM
from .decoder import HybridEncoder, RTDETRDecoder
from .losses import MAPPRPPLoss, box_cxcywh_to_xyxy


class MAPPRPPNet(nn.Module):
    """
    MAPPR++-Net: Complete Detection Network.
    
    Architecture:
    1. Backbone: ResNet18 with MARF++ blocks at S1-S4
    2. Shallow Feature Knowledge Complement (SKC):
       - STKE: Enhances shallow features
       - KAFF: Adaptive fusion of original and enhanced features
    3. CFPR: Cross-level feature progressive refinement
    4. AGRM: Attention-guided refinement at S2
    5. CARM: Context-aware refinement at S3 (with ASPP)
    6. Hybrid Encoder + RT-DETR Decoder
    
    Args:
        config: Model configuration dictionary
    """
    
    def __init__(self, config: dict = None):
        super(MAPPRPPNet, self).__init__()
        
        if config is None:
            config = {}
        
        model_config = config.get('MODEL', {})
        
        # Configuration flags
        self.use_stke = model_config.get('STKE', {}).get('ENABLED', True)
        self.use_kaff = model_config.get('KAFF', {}).get('ENABLED', True)
        self.use_cfpr = model_config.get('CFPR', {}).get('ENABLED', True)
        self.use_marfpp = model_config.get('MARFPP', {}).get('ENABLED', True)
        self.use_agrm = model_config.get('AGRM', {}).get('ENABLED', True)
        self.use_carm = model_config.get('CARM', {}).get('ENABLED', True)
        
        use_dps = model_config.get('MARFPP', {}).get('DPS_ENABLED', True)
        use_cpi = model_config.get('MARFPP', {}).get('CPI_ENABLED', True)
        
        # Backbone with MARF++
        self.backbone = ResNet18MARFPP(
            pretrained=model_config.get('PRETRAINED', True),
            use_marfpp=self.use_marfpp,
            use_dps=use_dps,
            use_cpi=use_cpi,
        )
        
        # Channel dimensions from ResNet18 stages
        s1_ch, s2_ch, s3_ch, s4_ch = 64, 128, 256, 512
        
        # STKE Module (after S1)
        if self.use_stke:
            self.stke = STKE(
                in_channels=s1_ch,
                reduction=model_config.get('STKE', {}).get('REDUCTION', 4)
            )
        
        # KAFF Module
        if self.use_kaff:
            self.kaff = KAFF(in_channels=s1_ch)
        
        # AGRM Module (at S2)
        if self.use_agrm:
            self.agrm = AGRM(
                in_channels=s2_ch,
                out_channels=128,
                reduction=model_config.get('AGRM', {}).get('REDUCTION', 16)
            )
        
        # CARM Module (at S3)
        if self.use_carm:
            self.carm = CARM(
                in_channels=s3_ch,
                out_channels=256,
                aspp_rates=model_config.get('CARM', {}).get('ASPP_RATES', [1, 6, 12, 18]),
                reduction=model_config.get('CARM', {}).get('REDUCTION', 16)
            )
        
        # CFPR Module
        if self.use_cfpr:
            self.cfpr = CFPR(
                in_channels_s1=s1_ch,
                in_channels_s2=128 if self.use_agrm else s2_ch,
                in_channels_s3=256 if self.use_carm else s3_ch,
                lrsa_dim=model_config.get('CFPR', {}).get('LRSA_DIM', 64),
                repc3_dim=model_config.get('CFPR', {}).get('REPC3_DIM', 128),
            )
        
        # Hybrid Encoder
        encoder_channels = [s1_ch, 128 if self.use_agrm else s2_ch,
                           256 if self.use_carm else s3_ch, s4_ch]
        self.encoder = HybridEncoder(
            in_channels_list=encoder_channels,
            hidden_dim=256,
            num_layers=model_config.get('DECODER', {}).get('NUM_ENCODER_LAYERS', 2),
            num_heads=model_config.get('DECODER', {}).get('NUM_HEADS', 8),
            dim_feedforward=model_config.get('DECODER', {}).get('DIM_FEEDFORWARD', 1024),
            dropout=model_config.get('DECODER', {}).get('DROPOUT', 0.1)
        )
        
        # RT-DETR Decoder
        decoder_config = model_config.get('DECODER', {})
        self.decoder = RTDETRDecoder(
            num_classes=config.get('DATASET', {}).get('NUM_CLASSES', 1),
            hidden_dim=256,
            num_queries=decoder_config.get('NUM_QUERIES', 300),
            num_decoder_layers=decoder_config.get('NUM_DECODER_LAYERS', 3),
            num_heads=decoder_config.get('NUM_HEADS', 8),
            dim_feedforward=decoder_config.get('DIM_FEEDFORWARD', 1024),
            dropout=decoder_config.get('DROPOUT', 0.1)
        )
        
        # Loss function
        train_config = config.get('TRAIN', {})
        self.criterion = MAPPRPPLoss(
            num_classes=config.get('DATASET', {}).get('NUM_CLASSES', 1),
            alpha=train_config.get('ALPHA', 1.0),
            beta=train_config.get('BETA', 1.0),
            lambda_l1=train_config.get('LAMBDA_L1', 5.0),
            lambda_giou=train_config.get('LAMBDA_GIOU', 2.0),
        )
        
        self._print_module_status()
    
    def _print_module_status(self):
        """Print which modules are enabled."""
        print("\n" + "="*60)
        print("MAPPR++-Net Configuration")
        print("="*60)
        print(f"  MARF++ Backbone: {'✓' if self.use_marfpp else '✗'}")
        print(f"  STKE Module:     {'✓' if self.use_stke else '✗'}")
        print(f"  KAFF Module:     {'✓' if self.use_kaff else '✗'}")
        print(f"  CFPR Module:     {'✓' if self.use_cfpr else '✗'}")
        print(f"  AGRM Module:     {'✓' if self.use_agrm else '✗'}")
        print(f"  CARM Module:     {'✓' if self.use_carm else '✗'}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  Total Parameters:    {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Model Size:          {total_params * 4 / 1024 / 1024:.1f} MB")
        print("="*60 + "\n")
    
    def forward(self, images: torch.Tensor, targets: List[Dict] = None) -> Dict:
        """
        Forward pass of MAPPR++-Net.
        
        Args:
            images: Input images [B, 3, H, W]
            targets: List of target dicts (for training)
            
        Returns:
            Dictionary with predictions and optionally losses
        """
        # 1. Backbone feature extraction
        s1, s2, s3, s4 = self.backbone(images)
        
        # 2. Shallow Feature Knowledge Complement (SKC)
        if self.use_stke and self.use_kaff:
            # STKE enhances shallow features
            f_stke = self.stke(s1)
            # KAFF fuses original and enhanced features
            f_kaff = self.kaff(s1, f_stke)
        else:
            f_kaff = s1
        
        # 3. AGRM at Stage S2
        if self.use_agrm:
            s2_refined, c1 = self.agrm(s2)
        else:
            s2_refined = s2
            c1 = s2
        
        # 4. CARM at Stage S3
        if self.use_carm:
            c2 = self.carm(s3)
        else:
            c2 = s3
        
        # 5. CFPR: Cross-Feature Progressive Refinement
        if self.use_cfpr:
            l_out, fc3 = self.cfpr(f_kaff, c1, c2)
            # Use refined features
            features = [f_kaff, c1, c2, s4]
        else:
            features = [f_kaff, s2_refined, s3, s4]
        
        # 6. Hybrid Encoder
        memory, fused_features = self.encoder(features)
        
        # 7. RT-DETR Decoder
        pred_logits, pred_boxes = self.decoder(memory)
        
        outputs = {
            'pred_logits': pred_logits,  # [B, Q, C+1]
            'pred_boxes': pred_boxes,     # [B, Q, 4]
        }
        
        # Compute losses if targets provided
        if targets is not None:
            losses = self.criterion(
                pred_logits, pred_boxes, targets
            )
            outputs['losses'] = losses
        
        return outputs
    
    def get_detections(self, outputs: Dict, conf_thresh: float = 0.3,
                       nms_thresh: float = 0.5, img_size: int = 640):
        """
        Post-process model outputs to get detections.
        
        Args:
            outputs: Model output dictionary
            conf_thresh: Confidence threshold
            nms_thresh: NMS IoU threshold
            img_size: Image size for scaling boxes
            
        Returns:
            List of detections per image
        """
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        B = pred_logits.shape[0]
        all_detections = []
        
        for b in range(B):
            probs = pred_logits[b].softmax(-1)  # [Q, C+1]
            scores, labels = probs[:, :-1].max(-1)  # Exclude background
            
            # Filter by confidence
            keep = scores > conf_thresh
            scores = scores[keep]
            labels = labels[keep]
            boxes = pred_boxes[b, keep]
            
            # Convert to xyxy format
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            boxes_xyxy = boxes_xyxy * img_size  # Scale to image size
            
            # NMS
            if len(scores) > 0:
                keep_indices = self._nms(boxes_xyxy, scores, nms_thresh)
                scores = scores[keep_indices]
                labels = labels[keep_indices]
                boxes_xyxy = boxes_xyxy[keep_indices]
            
            detections = {
                'scores': scores.cpu(),
                'labels': labels.cpu(),
                'boxes': boxes_xyxy.cpu(),
            }
            all_detections.append(detections)
        
        return all_detections
    
    @staticmethod
    def _nms(boxes, scores, threshold):
        """Non-Maximum Suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0].item()
            keep.append(i)
            
            xx1 = torch.max(x1[order[1:]], x1[i])
            yy1 = torch.max(y1[order[1:]], y1[i])
            xx2 = torch.min(x2[order[1:]], x2[i])
            yy2 = torch.min(y2[order[1:]], y2[i])
            
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            iou = inter / (areas[order[1:]] + areas[i] - inter)
            
            mask = iou <= threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long)