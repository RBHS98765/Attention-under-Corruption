"""Week 2 smoke test: all four model variants, all required checks.

Covers Part D (11 tests per variant), Part E (attention sanity checks), and
Part F (one-batch training smoke test) of the Week 2 spec.

Exits with a non-zero code if ANY test fails. Writes a structured result file
to week2_smoke_test_results.json next to this script.
"""
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import build_model, CONFIG
from models.adapters import count_parameters

VARIANTS = ["none", "se", "bam", "cbam"]
INPUT_SHAPES = [(2, 3, 32, 32), (1, 3, 32, 32), (4, 3, 32, 32)]
SEED = CONFIG["random_seed"]

RESULTS = {
    "variants": {}, "attention_sanity": {}, "one_batch": {}, "ok": True,
    "known_warnings": [
        "Upstream BAM ChannelGate contains BatchNorm1d after global pooling; a singleton B=1 training-mode forward raises ValueError because batch statistics cannot be estimated. Shape probes use eval mode for B=1, while the optimizer smoke test uses B=8 and passes."
    ],
}


def record(variant, name, ok, detail=""):
    RESULTS["variants"].setdefault(variant, {})[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        RESULTS["ok"] = False
        print(f"  FAIL [{variant}] {name}: {detail}")


def run_variant_tests(variant):
    """Part D: the 11 required tests for one variant."""
    torch.manual_seed(SEED)
    model = build_model(attention_type=variant)

    # 1. construction
    try:
        model = build_model(attention_type=variant)
        record(variant, "construct", True)
    except Exception as e:
        record(variant, "construct", False, repr(e))
        return

    # 2-3. forward + output shape for every input shape.
    # Use eval mode for shape probes, including B=1: the verified upstream BAM
    # implementation contains BatchNorm1d in its pooled channel gate, which
    # cannot estimate batch statistics from a singleton training batch.
    model.eval()
    for shape in INPUT_SHAPES:
        x = torch.randn(*shape)
        try:
            with torch.no_grad():
                out = model(x)
            ok = out.shape == (shape[0], 10)
            record(variant, f"forward_shape_{shape}", ok, f"got {tuple(out.shape)}")
        except Exception as e:
            record(variant, f"forward_shape_{shape}", False, repr(e))

    # 4. finite output
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    record(variant, "finite_output", torch.isfinite(out).all().item(), "")

    # 5. training mode
    model.train()
    record(variant, "training_mode", model.training, "")

    # 6. backward pass
    model.zero_grad()
    loss = model(x).pow(2).mean()
    try:
        loss.backward()
        record(variant, "backward", True)
    except Exception as e:
        record(variant, "backward", False, repr(e))

    # 7. finite gradient
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    ok = all(g is not None for g in grads) and all(torch.isfinite(g).all().item() for g in grads)
    record(variant, "finite_gradient", ok,
           f"{sum(g is None for g in grads)} missing / {sum(not torch.isfinite(g).all().item() for g in grads if g is not None)} non-finite")

    # 8. evaluation mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    record(variant, "eval_mode", (model.training is False) and torch.isfinite(out_eval).all().item(), "")

    # 9. state dict save/load
    sd = model.state_dict()
    model2 = build_model(attention_type=variant)
    try:
        model2.load_state_dict(sd)
        model.eval(); model2.eval()
        with torch.no_grad():
            a = model(x); b = model2(x)
        record(variant, "state_dict_save_load", torch.allclose(a, b), "")
    except Exception as e:
        record(variant, "state_dict_save_load", False, repr(e))

    # 10. CPU device
    record(variant, "cpu_device", next(model.parameters()).device.type == "cpu",
           str(next(model.parameters()).device))

    # 11. parameter accounting (backbone identity checked in detail by
    #     test_parameter_accounting.py; here we just confirm it is consistent)
    total = count_parameters(model)
    record(variant, "parameter_accounting", total > 0, f"total={total}")


def run_attention_sanity():
    """Part E: attention-specific sanity checks."""
    torch.manual_seed(SEED)
    x = torch.randn(2, 3, 32, 32)

    for variant in ("se", "bam", "cbam"):
        model = build_model(attention_type=variant)
        model.train()
        attns = model.attention_modules()
        # shape preservation: attention output must match input shape
        shapes_ok = True
        for att in attns:
            # feed a tensor of the block's channel count
            c = getattr(att, "gate_channel", None) or getattr(att, "channels", None)
            if c is None:
                c = getattr(att, "gate_channels", None)
            probe = torch.randn(2, c, 16, 16)
            out = att(probe)
            if out.shape != probe.shape:
                shapes_ok = False
        record(variant, "attention_shape_preserved", shapes_ok, "")

        # no NaN/Inf in attention outputs
        finite_ok = True
        for att in attns:
            c = getattr(att, "gate_channel", None) or getattr(att, "channels", None) or getattr(att, "gate_channels", None)
            out = att(torch.randn(2, c, 16, 16))
            if not torch.isfinite(out).all().item():
                finite_ok = False
        record(variant, "attention_no_nan_inf", finite_ok, "")

        # gradients reach attention parameters
        model.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        attn_params = [p for att in attns for p in att.parameters() if p.requires_grad]
        grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in attn_params)
        record(variant, "attention_gradients_reach", grads_ok,
               f"{sum(p.grad is None for p in attn_params)}/{len(attn_params)} missing grad")

        # attention params are included in model.parameters()
        model_params = {id(p) for p in model.parameters()}
        included = all(id(p) in model_params for p in attn_params)
        record(variant, "attention_in_model_parameters", included, "")

    # identity baseline contains no attention parameters
    model = build_model(attention_type="none")
    attn_params = [p for att in model.attention_modules() for p in att.parameters()]
    record("none", "identity_no_attention_params", len(attn_params) == 0,
           f"{len(attn_params)} params found")


def run_one_batch_training():
    """Part F: one-batch training smoke test per variant."""
    torch.manual_seed(SEED)
    for variant in VARIANTS:
        model = build_model(attention_type=variant)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        opt.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        after = [p.detach() for p in model.parameters() if p.requires_grad]
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        ok = torch.isfinite(loss).item() and changed
        RESULTS["one_batch"][variant] = {
            "loss": float(loss.item()), "loss_finite": bool(torch.isfinite(loss).item()),
            "params_changed": bool(changed), "ok": bool(ok),
        }
        if not ok:
            RESULTS["ok"] = False
            print(f"  FAIL [one_batch:{variant}] loss={loss.item():.4f} changed={changed}")
        else:
            print(f"  ok   [one_batch:{variant}] loss={loss.item():.4f} params_changed={changed}")


def main():
    print("=== Week 2 smoke test ===")
    print(f"seed={SEED}, variants={VARIANTS}, inputs={INPUT_SHAPES}")
    for v in VARIANTS:
        print(f"-- variant {v} --")
        run_variant_tests(v)
    print("-- attention sanity (Part E) --")
    run_attention_sanity()
    print("-- one-batch training smoke test (Part F) --")
    run_one_batch_training()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week2_smoke_test_results.json")
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)

    if RESULTS["ok"]:
        print("\nWEEK 2 SMOKE TEST: ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("\nWEEK 2 SMOKE TEST: FAILURES PRESENT")
        sys.exit(1)


if __name__ == "__main__":
    main()
