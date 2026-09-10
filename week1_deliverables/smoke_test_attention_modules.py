#!/usr/bin/env python3
"""smoke_test_attention_modules.py

Week 1 smoke tests for the official BAM and CBAM modules from
https://github.com/Jongchan/attention-module (commit 459efad0).

Tests: import, constructor, forward passes on multiple shapes, NaN/Inf checks,
dtype, training-mode + backward (gradient) checks, eval-mode no_grad check,
state-dict save/load equivalence, optional compatibility checks.

Note on BAM: the pristine upstream MODELS/bam.py raises NameError at
construction ("name 'gate_activation' is not defined") because ChannelGate.__init__
references an undefined variable. Upstream files are NOT modified. For the
forward/backward/state-dict tests we import a patched COPY (only the offending
line removed) from a temp directory, and this is recorded in the report.

Note on channel counts: BAM(gate_channel=C) and CBAM(gate_channels=C) expect an
input tensor with C channels; the module is instantiated per input shape with
the matching channel count.

Run:  python3 smoke_test_attention_modules.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import warnings

import torch

REPO_DIR = os.environ.get("ATTENTION_MODULE_DIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "attention-module"))
MODELS_DIR = os.path.join(REPO_DIR, "MODELS")
RESULTS = {}

def log(msg):
    print(msg, flush=True)

def import_pristine():
    sys.path.insert(0, MODELS_DIR)
    import bam as bam_mod
    import cbam as cbam_mod
    return bam_mod, cbam_mod

def load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def make_patched_bam_copy():
    src = os.path.join(MODELS_DIR, "bam.py")
    tmpdir = tempfile.mkdtemp(prefix="bam_patched_")
    with open(src) as f:
        content = f.read()
    bad = "        self.gate_activation = gate_activation\n"
    assert bad in content, "expected offending line not found"
    patched = content.replace(bad, "")
    dst = os.path.join(tmpdir, "bam.py")
    with open(dst, "w") as f:
        f.write(patched)
    mod = load_module_from_file("bam_patched", dst)
    return mod, "removed 'self.gate_activation = gate_activation' (undefined variable)", dst

def finite(x):
    return bool(torch.isfinite(x).all().item())

def make_input(shape, requires_grad=False):
    return torch.randn(*shape, requires_grad=requires_grad)

def forward_test(name, factory, shapes):
    out = {"shapes": {}}
    for shape in shapes:
        mod = factory(shape[1])
        x = make_input(shape)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with torch.no_grad():
                    y = mod(x)
            warns = [str(wi.message)[:140] for wi in w]
            out["shapes"][str(shape)] = {
                "output_shape": list(y.shape), "shape_preserved": tuple(y.shape) == tuple(shape),
                "finite": finite(y), "dtype": str(y.dtype), "dtype_ok": y.dtype == torch.float32,
                "device_moved": y.device != x.device, "warnings": warns}
            log(f"  [{name}] fwd {shape} -> {tuple(y.shape)} shape_ok={tuple(y.shape)==tuple(shape)} finite={finite(y)} dtype={y.dtype} moved={y.device!=x.device} warns={len(warns)}")
        except Exception as e:
            out["shapes"][str(shape)] = {"error": f"{type(e).__name__}: {e}"}
            log(f"  [{name}] fwd {shape} FAILED: {type(e).__name__}: {e}")
    return out

def backward_test(name, factory, shapes):
    out = {}
    for shape in shapes:
        mod = factory(shape[1]); mod.train()
        x = make_input(shape, requires_grad=True)
        try:
            y = mod(x)
            loss = y.mean()
            loss.backward()
            grad_ok = x.grad is not None and finite(x.grad)
            params_with_grad = sum(1 for p in mod.parameters() if p.requires_grad and p.grad is not None and finite(p.grad))
            params_total = sum(1 for p in mod.parameters() if p.requires_grad)
            out[str(shape)] = {"loss": float(loss.item()), "input_grad_exists": x.grad is not None,
                               "input_grad_finite": grad_ok, "params_with_grad": params_with_grad, "params_total": params_total}
            log(f"  [{name}] backward {shape}: loss={loss.item():.6f} input_grad={x.grad is not None} params_grad={params_with_grad}/{params_total}")
        except Exception as e:
            out[str(shape)] = {"error": f"{type(e).__name__}: {e}"}
            log(f"  [{name}] backward {shape} FAILED: {type(e).__name__}: {e}")
    return out

def eval_test(name, factory, shapes):
    out = {}
    for shape in shapes:
        mod = factory(shape[1]); mod.eval()
        x = make_input(shape)
        try:
            with torch.no_grad():
                y = mod(x)
            out[str(shape)] = {"output_shape": list(y.shape), "shape_preserved": tuple(y.shape) == tuple(shape), "finite": finite(y)}
            log(f"  [{name}] eval {shape} -> {tuple(y.shape)} finite={finite(y)}")
        except Exception as e:
            out[str(shape)] = {"error": f"{type(e).__name__}: {e}"}
    return out

def state_dict_test(name, factory, shapes):
    out = {}
    shape = shapes[0]
    x = make_input(shape)
    try:
        m1 = factory(shape[1]); m2 = factory(shape[1])
        m1.eval(); m2.eval()
        with torch.no_grad():
            y1 = m1(x)
        fd, path = tempfile.mkstemp(suffix=".pth"); os.close(fd)
        torch.save(m1.state_dict(), path)
        m2.load_state_dict(torch.load(path, map_location="cpu"))
        with torch.no_grad():
            y2 = m2(x)
        max_diff = float((y1 - y2).abs().max().item())
        out = {"state_dict_tensors": len(m1.state_dict()), "max_abs_diff": max_diff, "equivalent": max_diff < 1e-6}
        log(f"  [{name}] state_dict: {len(m1.state_dict())} tensors, max_abs_diff={max_diff:.3e} equivalent={max_diff < 1e-6}")
        os.remove(path)
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}"}
        log(f"  [{name}] state_dict FAILED: {type(e).__name__}: {e}")
    return out

def compat_test(name, factory, shapes):
    out = {}
    extra = [(1, 64, 32, 32), (1, 64, 4, 4), (2, 64, 31, 31)]
    for shape in extra:
        mod = factory(shape[1]); mod.eval()
        try:
            x = make_input(shape)
            with torch.no_grad():
                y = mod(x)
            out[str(shape)] = {"output_shape": list(y.shape), "ok": tuple(y.shape) == tuple(shape)}
            log(f"  [{name}] compat {shape} -> {tuple(y.shape)}")
        except Exception as e:
            out[str(shape)] = {"error": f"{type(e).__name__}: {e}"}
            log(f"  [{name}] compat {shape} FAILED: {type(e).__name__}: {e}")
    try:
        mod = factory(shapes[0][1]); mod.eval()
        ts = torch.jit.trace(mod, make_input(shapes[0]))
        out["torchscript_trace"] = "ok"
        log(f"  [{name}] torchscript trace ok")
    except Exception as e:
        out["torchscript_trace"] = f"{type(e).__name__}: {e}"
        log(f"  [{name}] torchscript trace FAILED (non-blocking): {type(e).__name__}: {e}")
    return out

def main():
    log("=== smoke_test_attention_modules.py ===")
    log("PyTorch: " + torch.__version__ + " | CUDA available: " + str(torch.cuda.is_available()))
    log("MODELS dir: " + MODELS_DIR)
    log("F.sigmoid exists: " + str(hasattr(torch.nn.functional, "sigmoid")))

    SHAPES = [(2, 64, 32, 32), (2, 128, 16, 16), (2, 256, 8, 8)]

    try:
        bam_pristine, cbam_pristine = import_pristine()
        RESULTS["import"] = {"bam": "ok", "cbam": "ok"}
        log("IMPORT: bam ok, cbam ok")
    except Exception as e:
        RESULTS["import"] = {"error": f"{type(e).__name__}: {e}"}
        log("IMPORT FAILED: " + repr(e))
        print(json.dumps(RESULTS, indent=2, default=str)); return 1

    # ---- BAM ----
    log("--- BAM ---")
    bam_r = {"import": "ok"}
    try:
        bam_pristine.BAM(gate_channel=64)
        bam_r["pristine_constructor"] = "ok"
        bam_mod = bam_pristine
        bam_r["patch_used"] = None
    except Exception as e:
        bam_r["pristine_constructor"] = f"{type(e).__name__}: {e}"
        log("BAM pristine constructor FAILED: " + repr(e))
        bam_mod, patch_desc, patch_path = make_patched_bam_copy()
        bam_r["patch_used"] = patch_desc
        bam_r["patch_path"] = patch_path
        log("Using patched BAM copy: " + patch_desc)
    try:
        bam = bam_mod.BAM(gate_channel=64)
        bam_r["constructor"] = "ok"
    except Exception as e:
        bam_r["constructor"] = f"{type(e).__name__}: {e}"
        RESULTS["bam"] = bam_r
        print(json.dumps(RESULTS, indent=2, default=str)); return 1
    bam_factory = lambda C: bam_mod.BAM(gate_channel=C)
    bam_r["forward"] = forward_test("BAM", bam_factory, SHAPES)
    bam_r["backward"] = backward_test("BAM", bam_factory, SHAPES)
    bam_r["eval"] = eval_test("BAM", bam_factory, SHAPES)
    bam_r["state_dict"] = state_dict_test("BAM", bam_factory, SHAPES)
    bam_r["compat"] = compat_test("BAM", bam_factory, SHAPES)
    RESULTS["bam"] = bam_r

    # ---- CBAM ----
    log("--- CBAM ---")
    cbam_r = {"import": "ok"}
    try:
        cbam = cbam_pristine.CBAM(gate_channels=64)
        cbam_r["constructor"] = "ok"
        log("CBAM constructor ok")
    except Exception as e:
        cbam_r["constructor"] = f"{type(e).__name__}: {e}"
        RESULTS["cbam"] = cbam_r
        print(json.dumps(RESULTS, indent=2, default=str)); return 1
    cbam_factory = lambda C: cbam_pristine.CBAM(gate_channels=C)
    cbam_r["forward"] = forward_test("CBAM", cbam_factory, SHAPES)
    cbam_r["backward"] = backward_test("CBAM", cbam_factory, SHAPES)
    cbam_r["eval"] = eval_test("CBAM", cbam_factory, SHAPES)
    cbam_r["state_dict"] = state_dict_test("CBAM", cbam_factory, SHAPES)
    cbam_r["compat"] = compat_test("CBAM", cbam_factory, SHAPES)
    RESULTS["cbam"] = cbam_r

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_test_results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(json.dumps(RESULTS, indent=2, default=str))
    return 0

if __name__ == "__main__":
    sys.exit(main())
