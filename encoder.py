"""
RT-DETR Encoder for MAPPR++-Net.
Multi-scale feature encoding with deformable attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleDeformableAttention(nn.Module):
    """Simplified multi-scale deformable attention for encoder."""

    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.sampling_offsets.weight)
        nn.init.xavier_uniform_(self.attention_weights.weight)

    def forward(self, query, reference_points, value_list, spatial_shapes):
        """
        Simplified deformable attention forward.
        """
        B, N, C = query.shape
        value = torch.cat(value_list, dim=1)

        # Value projection
        value = self.value_proj(value)

        # Simple multi-head attention as fallback
        query_proj = self.output_proj.weight[:C, :C]  # Simplified
        scale = self.head_dim ** -0.5

        # Reshape for multi-head attention
        q = query.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = value.reshape(B, value.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, v.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.output_proj(out)

        return out


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer with self-attention and FFN."""

    def __init__(self, embed_dim=256, num_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, pos=None):
        q = k = src if pos is None else src + pos
        src2, _ = self.self_attn(q, k, value=src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class RTEncoder(nn.Module):
    """
    RT-DETR Encoder that processes multi-scale features.
    """

    def __init__(self, embed_dim=256, num_layers=4, num_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src, pos=None):
        """Process flattened feature sequence through encoder layers."""
        for layer in self.layers:
            src = layer(src, pos)
        src = self.norm(src)
        return src