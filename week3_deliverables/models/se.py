"""Squeeze-and-Excitation block (SE), implemented from the SE paper equations.

Reference: J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation Networks", CVPR 2018.
The block follows the paper's two-step formulation:

  Squeeze:    z_c = F_sq(u_c) = (1/(H*W)) * sum_{i,j} u_c(i, j)     (global avg pool)
  Excitation: s = F_ex(z, W) = sigma( W2 * ReLU(W1 * z) )           (bottleneck MLP)
  Scale:      x_tilde_c = F_scale(u_c, s_c) = s_c * u_c             (channel rescale)

where W1 in R^{(C/r) x C}, W2 in R^{C x (C/r)}, sigma is the sigmoid gate, and
r is the reduction ratio. Input [B, C, H, W] -> output [B, C, H, W].

Reduction ratio (documented, Part A2):
  r = 16 is chosen for this experiment as a standard, commonly used value. It
  is a configuration choice, NOT claimed to be uniquely recovered from the SE
  paper (the paper evaluates several ratios). Hidden width:
      hidden_channels = max(C // r, 1)
  so tiny channel counts still produce a valid bottleneck.

Activation: ReLU in the bottleneck (as in the paper). Initialization: default
PyTorch Linear initialization (no custom init). No bias handling changes.
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
        self.fc1 = nn.Linear(channels, hidden)   # W1: C -> C/r
        self.fc2 = nn.Linear(hidden, channels)  # W2: C/r -> C
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Squeeze: global average pooling -> [B, C]
        z = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation: bottleneck MLP -> sigmoid gate
        s = self.fc2(self.relu(self.fc1(z)))
        s = self.sigmoid(s).view(b, c, 1, 1)
        # Scale: channel-wise rescaling (shape preserved)
        return x * s
