"""Week 2 parameter-accounting tests.

Verifies, for every variant:
  - total, trainable, non-trainable, backbone-only, attention-only, classifier
    parameter counts are reported consistently;
  - backbone-only parameter counts are EXACTLY identical across all variants;
  - total-parameter differences are attributable only to the attention blocks;
  - each attention block is configured for its own feature-map channel count.

Runnable standalone (python tests/test_parameter_accounting.py) or under
pytest. Exits non-zero if any check fails.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import build_model, CONFIG
from models.adapters import count_parameters

VARIANTS = ["none", "se", "bam", "cbam"]
SEED = CONFIG["random_seed"]

_FAILURES = []


def _check(name, cond, detail=""):
    if not cond:
        _FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}: {detail}")
    else:
        print(f"  ok   {name}")


def account(model):
    """Return a dict of parameter counts for a model variant."""
    total = count_parameters(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    attention = sum(count_parameters(m) for m in model.attention_modules())
    classifier = count_parameters(model.classifier)
    backbone_only = total - attention          # conv/BN stem+stages+classifier
    backbone_conv = backbone_only - classifier  # conv/BN part only
    n_attn_blocks = len(model.attention_modules())
    return {
        "total": total, "trainable": trainable, "non_trainable": non_trainable,
        "backbone_only": backbone_only, "attention": attention,
        "classifier": classifier, "backbone_conv": backbone_conv,
        "n_attention_blocks": n_attn_blocks,
    }


def run_parameter_accounting():
    torch.manual_seed(SEED)
    accounts = {}
    for v in VARIANTS:
        model = build_model(attention_type=v)
        accounts[v] = account(model)

    # backbone-only must be exactly identical across variants
    bb = {v: a["backbone_only"] for v, a in accounts.items()}
    _check("backbone params identical",
           len(set(bb.values())) == 1,
           f"backbone_only differ: {bb}")

    # plain baseline has zero attention params
    _check("plain has zero attention params",
           accounts["none"]["attention"] == 0,
           f"plain attention={accounts['none']['attention']}")

    # attention-enabled variants have > 0 attention params
    for v in ("se", "bam", "cbam"):
        _check(f"{v} has attention params", accounts[v]["attention"] > 0,
               f"attention={accounts[v]['attention']}")

    # total difference from plain is exactly the attention params of that variant
    base_total = accounts["none"]["total"]
    for v in ("se", "bam", "cbam"):
        diff = accounts[v]["total"] - base_total
        _check(f"{v} total diff == attention params",
               diff == accounts[v]["attention"],
               f"diff={diff} vs attention={accounts[v]['attention']}")

    # non-trainable params: BatchNorm in the backbone is trainable by default,
    # so non-trainable should be 0 for every variant (nothing frozen).
    for v in VARIANTS:
        _check(f"{v} non-trainable == 0", accounts[v]["non_trainable"] == 0,
               f"non_trainable={accounts[v]['non_trainable']}")

    # every attention-enabled variant has one attention block per residual block (4)
    for v in VARIANTS:
        _check(f"{v} n_attention_blocks==4", accounts[v]["n_attention_blocks"] == 4,
               f"n={accounts[v]['n_attention_blocks']}")

    # per-block attention channel counts are correct (64, 64, 128, 256)
    model = build_model(attention_type="se")
    attns = model.attention_modules()
    expected = [64, 64, 128, 256]
    for i, (att, c) in enumerate(zip(attns, expected)):
        _check(f"se block {i} channels=={c}", att.channels == c,
               f"got {getattr(att, 'channels', None)}")

    return accounts


def main():
    print("=== test_parameter_accounting.py ===")
    accounts = run_parameter_accounting()
    print("\nParameter table:")
    hdr = f"{'variant':<6} {'total':>9} {'trainable':>9} {'non-train':>9} {'backbone':>9} {'attn':>7} {'classifier':>10}"
    print(hdr)
    for v in VARIANTS:
        a = accounts[v]
        print(f"{v:<6} {a['total']:>9} {a['trainable']:>9} {a['non_trainable']:>9} "
              f"{a['backbone_only']:>9} {a['attention']:>7} {a['classifier']:>10}")
    if _FAILURES:
        print(f"\n{len(_FAILURES)} FAILURES")
        sys.exit(1)
    print("\nALL PARAMETER-ACCOUNTING TESTS PASSED")


if __name__ == "__main__":
    main()
