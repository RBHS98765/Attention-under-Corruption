"""Matched CNN backbone for CIFAR-10 with a pluggable attention slot.

Design (documented before testing, Part A / Part G of the Week 2 spec):

  Input:  [B, 3, 32, 32]  (CIFAR-10, 3-channel, no ImageNet assumptions)
  Stem:   Conv2d(3, 64, 3, stride=1, padding=1) -> BN -> ReLU   (32x32 kept)
  Stage 1: two residual blocks at 64 channels (no downsampling)  (32x32)
  Stage 2: one residual block 64 -> 128, first conv stride 2     (16x16)
  Stage 3: one residual block 128 -> 256, first conv stride 2    (8x8)
  Head:   Global average pooling -> Linear(256, 10)

  Total residual blocks: 4. No initial 7x7 conv, no initial stride 2, no
  ImageNet-sized pooling. The exact same convolutional structure is used in
  every model variant; only the attention slot differs.

Attention insertion point (A2):
  After the residual block's convolutional feature transformation (conv2 + bn2)
  and BEFORE the residual addition. This exact location is used for SE, BAM,
  and CBAM alike. Attention is applied in every residual block, at its own
  channel count (64, 64, 128, 256) per Part B3.

The backbone accepts attention_type in {None, "se", "bam", "cbam"} and is
built once per variant via build_model(); the backbone is never duplicated.
"""
import torch
import torch.nn as nn

from .adapters import build_attention

# ---------------------------------------------------------------------------
# Configuration freeze (Part G). The architecture is fixed here and is NOT
# changed per attention method.
# ---------------------------------------------------------------------------
CONFIG = {
    "backbone_name": "MatchedBackbone",
    "input_size": [3, 32, 32],
    "num_classes": 10,
    "channel_widths": [64, 64, 128, 256],
    "num_residual_blocks": 4,
    "residual_blocks_per_stage": [2, 1, 1],
    "stem": "Conv2d(3,64,3,stride=1,padding=1) + BatchNorm2d + ReLU",
    "attention_insertion": "after_conv_transformation_before_residual_add (every block)",
    "attention_type": None,  # set per variant; None|'se'|'bam'|'cbam'
    "se_reduction": 16,
    "bam_settings": {"reduction_ratio": 16, "dilation_conv_num": 2, "dilation_val": 4},
    "cbam_settings": {"reduction_ratio": 16, "pool_types": ["avg", "max"], "no_spatial": False},
    "normalization": "BatchNorm2d",
    "activation": "ReLU",
    "classifier": "AdaptiveAvgPool2d(1) -> Linear(256, 10)",
    "parameter_counting_method": "sum(p.numel() for p in model.parameters())",
    "random_seed": 0,
}


class ResidualBlock(nn.Module):
    """Basic residual block with a pluggable attention slot.

    Attention is applied to the transformed features (after conv2+bn2) and
    before the residual addition, at the block's own channel count.
    """

    def __init__(self, in_channels, out_channels, stride=1, attention=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention if attention is not None else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)          # <-- attention insertion point
        out = out + identity              # residual addition
        out = self.relu(out)
        return out


class MatchedBackbone(nn.Module):
    """CIFAR-10 matched backbone with a pluggable attention slot.

    Args:
        attention_type: None | "se" | "bam" | "cbam".
        num_classes: number of output logits (default 10).
        se_reduction / bam_reduction / cbam_reduction: attention hyper-params.

    Forward:
        logits = model(x)                       -> [B, 10]
        logits, attention_info = model(x, return_attention=True)
    The default path is compatible with a standard PyTorch training loop.
    """

    def __init__(self, attention_type=None, num_classes=10,
                 se_reduction=16, bam_reduction=16, cbam_reduction=16):
        super().__init__()
        self.attention_type = (attention_type or "none").lower()
        self.num_classes = num_classes

        def att(channels):
            return build_attention(
                self.attention_type, channels,
                se_reduction=se_reduction,
                bam_reduction=bam_reduction,
                cbam_reduction=cbam_reduction,
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, attention=att(64)),
            ResidualBlock(64, 64, stride=1, attention=att(64)),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, attention=att(128)),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, attention=att(256)),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_attention=False):
        if return_attention:
            attention_info = []
        else:
            attention_info = None

        h = self.stem(x)
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                h = block(h)
                if return_attention:
                    attention_info.append(block.attention)

        h = self.gap(h).flatten(1)
        logits = self.classifier(h)
        if return_attention:
            return logits, attention_info
        return logits

    def attention_modules(self):
        """List of the attention submodules, one per residual block."""
        mods = []
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                mods.append(block.attention)
        return mods


def build_model(attention_type=None, num_classes=10,
                se_reduction=16, bam_reduction=16, cbam_reduction=16):
    """Factory: build one model variant from the shared backbone class."""
    return MatchedBackbone(
        attention_type=attention_type,
        num_classes=num_classes,
        se_reduction=se_reduction,
        bam_reduction=bam_reduction,
        cbam_reduction=cbam_reduction,
    )
