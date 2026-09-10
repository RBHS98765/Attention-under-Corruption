"""Week 2 matched-backbone package.

Exposes the matched backbone, the pluggable attention factory, and the
concrete attention blocks (SE, BAM, CBAM).
"""
from .backbone import MatchedBackbone, build_model, CONFIG
from .se import SEBlock
from .attention import BAM, CBAM, Identity
from .adapters import build_attention

__all__ = [
    "MatchedBackbone", "build_model", "CONFIG",
    "SEBlock", "BAM", "CBAM", "Identity", "build_attention",
]
