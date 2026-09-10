"""Week 2 model inspector.

Prints, for every variant:
  - the architecture (module tree, truncated);
  - the parameter-accounting table (total / trainable / non-trainable /
    backbone-only / attention / classifier);
  - estimated MACs for a [1, 3, 32, 32] input (conv + linear layers only).

Usage: python scripts/inspect_models.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import build_model, CONFIG
from models.adapters import count_parameters

VARIANTS = ["none", "se", "bam", "cbam"]


def count_macs(model, input_shape=(1, 3, 32, 32)):
    """Estimate MACs by hooking Conv2d/Linear and using real tensor shapes."""
    macs = {"conv": 0, "linear": 0}
    hooks = []

    def hook_fn(m, inp, out):
        if isinstance(m, torch.nn.Conv2d):
            n, c_out, h_out, w_out = out.shape
            k = m.kernel_size[0] * m.kernel_size[1]
            macs["conv"] += n * c_out * h_out * w_out * (m.in_channels // m.groups) * k
        elif isinstance(m, torch.nn.Linear):
            macs["linear"] += out.numel() * m.in_features

    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_shape))
    for h in hooks:
        h.remove()
    return macs


def main():
    print("=== Week 2 model inspector ===")
    print("CONFIG:", json_dumps_pretty(CONFIG))
    print()
    hdr = (f"{'variant':<6} {'total':>9} {'trainable':>9} {'non-train':>9} "
           f"{'backbone':>9} {'attn':>7} {'classifier':>10} {'MACs(conv+lin)':>16}")
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        model = build_model(attention_type=v)
        total = count_parameters(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        attention = sum(count_parameters(m) for m in model.attention_modules())
        classifier = count_parameters(model.classifier)
        backbone = total - attention
        macs = count_macs(model)
        total_macs = macs["conv"] + macs["linear"]
        print(f"{v:<6} {total:>9} {trainable:>9} {total - trainable:>9} "
              f"{backbone:>9} {attention:>7} {classifier:>10} {total_macs:>16,}")
    print()
    print("Architecture (plain variant as reference):")
    model = build_model(attention_type="none")
    print(model)


def json_dumps_pretty(obj):
    import json
    return json.dumps(obj, indent=2)


if __name__ == "__main__":
    main()
