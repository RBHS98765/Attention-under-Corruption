"""Squeeze-and-Excitation block (SE), implemented directly in PyTorch.

This is a fresh PyTorch implementation following the standard SE design:
global average pooling -> two-layer channel bottleneck -> ReLU -> sigmoid
gate -> channel-wise rescaling.

The reduction ratio r is a configuration choice for this experiment, not a
value claimed to be uniquely recovered from the original paper. If the channel
count is C and the reduction ratio is r, the hidden width is
    hidden_channels = max(C // r, 1)
so that tiny channel counts still produce a valid (non-empty) bottleneck.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Channel-wise Squeeze-and-Excitation block.

    Args:
        channels: number of input/output channels C of the feature map.
        reduction: reduction ratio r; hidden width is max(C // r, 1).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if reduction < 1:
            raise ValueError(f"reduction must be >= 1, got {reduction}")
        self.channels = channels
        self.reduction = reduction
        hidden = max(channels // reduction, 1)
        self.hidden = hidden
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: global average pooling over spatial dims -> [B, C]
        b, c, h, w = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation: bottleneck MLP -> sigmoid gate
        s = self.fc2(self.relu(self.fc1(s)))
        s = self.sigmoid(s).view(b, c, 1, 1)
        # Rescale: channel-wise multiplication (shape preserved)
        return x * s
