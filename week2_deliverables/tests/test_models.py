"""Week 2 model tests: construction, forward, backward, shapes, state dict.

Runnable standalone (python tests/test_models.py) or under pytest. Exits
non-zero if any check fails.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import build_model, CONFIG
from models.adapters import count_parameters

VARIANTS = ["none", "se", "bam", "cbam"]
INPUT_SHAPES = [(2, 3, 32, 32), (1, 3, 32, 32), (4, 3, 32, 32)]
SEED = CONFIG["random_seed"]

_FAILURES = []


def _check(name, cond, detail=""):
    if not cond:
        _FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}: {detail}")
    else:
        print(f"  ok   {name}")


def _make(variant):
    torch.manual_seed(SEED)
    return build_model(attention_type=variant)


def run_model_tests(variant):
    print(f"== variant: {variant} ==")
    model = _make(variant)

    # 1. construction
    _check("construct", model is not None, "model is None")

    # 2-3. forward + output shape for every input shape.
    # Use eval mode for shape probes, including B=1: the verified upstream BAM
    # implementation contains BatchNorm1d in its pooled channel gate, which
    # cannot estimate batch statistics from a singleton training batch.
    model.eval()
    for shape in INPUT_SHAPES:
        x = torch.randn(*shape)
        with torch.no_grad():
            out = model(x)
        _check(f"forward {shape}", out.shape == (shape[0], 10),
               f"got {tuple(out.shape)}")

    # 4. finite output
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    _check("finite output", torch.isfinite(out).all().item(), "non-finite logits")

    # 5. training mode
    model.train()
    _check("training mode", model.training, "not in train mode")

    # 6-7. backward + finite gradients
    model.zero_grad()
    loss = model(x).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    _check("backward ran", all(g is not None for g in grads),
           f"{sum(g is None for g in grads)} params missing grad")
    _check("finite gradients", all(torch.isfinite(g).all().item() for g in grads),
           "non-finite gradient")

    # 8. evaluation mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    _check("eval mode", model.training is False, "still training")
    _check("eval finite output", torch.isfinite(out_eval).all().item(), "non-finite")

    # 9. state dict save/load
    sd = model.state_dict()
    model2 = _make(variant)
    model2.load_state_dict(sd)
    model.eval(); model2.eval()
    with torch.no_grad():
        a = model(x); b = model2(x)
    _check("state_dict save/load", torch.allclose(a, b), "outputs differ after load")

    # 10. CPU device
    _check("cpu device", next(model.parameters()).device.type == "cpu",
           f"device={next(model.parameters()).device}")

    # 11. parameter accounting (see test_parameter_accounting.py for detail)
    total = count_parameters(model)
    _check("parameter accounting", total > 0, "zero total params")


def main():
    print("=== test_models.py ===")
    for v in VARIANTS:
        run_model_tests(v)
    if _FAILURES:
        print(f"\n{len(_FAILURES)} FAILURES")
        sys.exit(1)
    print("\nALL MODEL TESTS PASSED")


if __name__ == "__main__":
    main()
