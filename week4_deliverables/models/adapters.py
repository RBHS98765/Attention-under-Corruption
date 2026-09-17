"""Pluggable attention factory / adapter.

The backbone never hard-codes an attention module. It asks this factory for an
attention instance given the exact feature-map channel count of the stage, so
a 64-channel feature map gets an attention block configured for 64 channels,
a 128-channel map for 128, and a 256-channel map for 256.

Registry:
    None / "none" -> Identity()          (plain CNN baseline, zero params)
    "se"         -> SEBlock(channels, reduction=se_reduction)
    "bam"        -> BAM(channels, reduction_ratio=bam_reduction)
    "cbam"       -> CBAM(channels, reduction_ratio=cbam_reduction)
"""
import torch.nn as nn

from .se import SEBlock
from .attention import BAM, CBAM, Identity

ATTENTION_TYPES = ("none", "se", "bam", "cbam")


def build_attention(attention_type, channels, se_reduction=16, bam_reduction=16,
                    cbam_reduction=16, cbam_pool_types=("avg", "max"),
                    cbam_no_spatial=False):
    """Return an attention module for a feature map with ``channels`` channels.

    ``attention_type`` may be None, "none", "se", "bam", or "cbam".
    """
    t = (attention_type or "none").lower()
    if t not in ATTENTION_TYPES:
        raise ValueError(
            f"Unknown attention_type {attention_type!r}; expected one of "
            f"{ATTENTION_TYPES} (or None)."
        )
    if t == "none":
        return Identity()
    if t == "se":
        return SEBlock(channels, reduction=se_reduction)
    if t == "bam":
        return BAM(channels, reduction_ratio=bam_reduction)
    if t == "cbam":
        return CBAM(channels, reduction_ratio=cbam_reduction,
                    pool_types=cbam_pool_types, no_spatial=cbam_no_spatial)
    raise AssertionError("unreachable")


def count_parameters(module):
    """Total number of parameters in a module (trainable + non-trainable)."""
    return sum(p.numel() for p in module.parameters())
