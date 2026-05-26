"""
RT-DETR Decoder and Hybrid Encoder for MAPPR++-Net

Based on: "DETRs Beat YOLOs on Real-Time Object Detection"
(Zhao et al., CVPR 2024) - https://arxiv.org/abs/2304.08069
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MLP(nn.Module):
    """Simple multi-layer perceptron."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional cross-attention."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None):
        B, N, C = query.shape
        _, S, _ = key.shape
        
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn = attn + attn_mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer."""
    
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, query, memory, query_pos=None, memory_pos=None):
        # Self-attention
        q = query + query_pos if query_pos is not None else query
        q2 = self.self_attn(q, q, q)
        query = query + self.dropout1(q2)
        query = self.norm1(query)
        
        # Cross-attention
        q = query + query_pos if query_pos is not None else query
        k = memory + memory_pos if memory_pos is not None else memory
        q2 = self.cross_attn(q, k, memory)
        query = query + self.dropout2(q2)
        query = self.norm2(query)
        
        # FFN
        query = query + self.ffn(query)
        query = self.norm3(query)
        
        return query


class HybridEncoder(nn.Module):
    """
    Hybrid Encoder from RT-DETR.
    Fuses multi-scale features from backbone with AIFI attention.
    """
    
    def __init__(self, in_channels_list=[64, 128, 256, 512],
                 hidden_dim=256, num_layers=2, num_heads=8,
                 dim_feedforward=1024, dropout=0.1):
        super(HybridEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Feature projections
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        
        # Encoder layers (AIFI-style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Feature fusion convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels_list))
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from backbone [S1, S2, S3, S4]
            
        Returns:
            Encoded feature sequence for decoder
        """
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            p = proj(feat)
            projected.append(p)
        
        # Top-down feature fusion (FPN-style)
        fused = [None] * len(projected)
        fused[-1] = projected[-1]
        for i in range(len(projected) - 2, -1, -1):
            up = F.interpolate(fused[i + 1], size=projected[i].shape[2:],
                              mode='bilinear', align_corners=False)
            fused[i] = self.lateral_convs[i](projected[i] + up)
        
        # Flatten and concatenate for transformer encoding
        flattened = []
        for feat in fused:
            B, C, H, W = feat.shape
            flattened.append(feat.view(B, C, -1).permute(0, 2, 1))
        
        seq = torch.cat(flattened, dim=1)  # [B, N, C]
        
        # Transformer encoding
        encoded = self.encoder(seq)
        
        return encoded, fused


class RTDETRDecoder(nn.Module):
    """
    RT-DETR Detection Decoder.
    
    Args:
        num_classes: Number of object classes
        hidden_dim: Hidden dimension
        num_queries: Number of object queries
        num_decoder_layers: Number of decoder layers
        num_heads: Number of attention heads
        dim_feedforward: FFN dimension
        dropout: Dropout rate
    """
    
    def __init__(self, num_classes=1, hidden_dim=256, num_queries=300,
                 num_decoder_layers=3, num_heads=8,
                 dim_feedforward=1024, dropout=0.1):
        super(RTDETRDecoder, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads,
                                    dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)  # cx, cy, w, h
        
        self._init_weights()
    
    def _init_weights(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_head.bias[-1], bias_value)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, memory, memory_pos=None):
        """
        Args:
            memory: Encoded features [B, N, C]
            memory_pos: Positional encoding [B, N, C]
            
        Returns:
            Tuple of (class_predictions, bbox_predictions)
        """
        B = memory.shape[0]
        
        # Initialize queries
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Decode
        for layer in self.decoder_layers:
            query = layer(query, memory, query_pos=None, memory_pos=memory_pos)
        
        # Predict
        class_pred = self.class_head(query)   # [B, num_queries, num_classes+1]
        bbox_pred = self.bbox_head(query)      # [B, num_queries, 4]
        
        # Sigmoid for bbox (normalized)
        bbox_pred = bbox_pred.sigmoid()
        
        return class_pred, bbox_pred