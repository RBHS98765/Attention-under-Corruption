# Week 1 Attention-Module Smoke Test Report

Modules: BAM and CBAM from https://github.com/Jongchan/attention-module (commit 459efad0e05ee7dde50c41ca10a3d0800bc3792a, MIT license, last commit 2022-09-12).
Environment: Python 3.12.10, PyTorch 2.13.0+cpu, Torchvision 0.28.0+cpu, CPU only.
Script: smoke_test_attention_modules.py (run: python3 smoke_test_attention_modules.py). Raw output: smoke_test_results.json, smoke_test.log.

## 1. Import results (C1)

Both modules import cleanly from the pristine upstream files (no repository modification): bam ok, cbam ok.

## 2. Constructor details

BAM:
- Class BAM(gate_channel). Composed of ChannelGate(gate_channel, reduction_ratio=16, num_layers=1) and SpatialGate(gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4).
- ChannelGate: flatten -> Linear(gate_channel -> gate_channel//16) -> BN1d -> ReLU -> Linear(... -> gate_channel); global average pool over the full spatial extent.
- SpatialGate: 1x1 conv reduce to gate_channel//16, then 2 dilated 3x3 convs (dilation=4, padding=4), then 1x1 conv to 1 channel.
- Forward: att = 1 + sigmoid(channel_att * spatial_att); output = att * input. Output shape == input shape.

CBAM:
- Class CBAM(gate_channels, reduction_ratio=16, pool_types=['avg','max'], no_spatial=False).
- ChannelGate: MLP (Linear -> gate_channels//16 -> ReLU -> Linear) applied to avg and max pooled features, summed, sigmoid, broadcast multiply.
- SpatialGate: ChannelPool (max + mean across channels) -> 7x7 conv (padding 3) -> sigmoid -> multiply.
- Output shape == input shape.

## 3. BAM upstream bug (documented, not silently fixed)

The pristine MODELS/bam.py fails at construction with:
NameError: name 'gate_activation' is not defined
Cause: ChannelGate.__init__ contains the line "self.gate_activation = gate_activation" but gate_activation is neither a parameter nor defined anywhere in the module. The attribute is never used elsewhere, so the line is dead code. This is a genuine bug in the official file (the repo README itself warns "There may be minor errors in the training code").

Handling: upstream files were NOT modified. The smoke test imports the pristine module for the import test, then, for the forward/backward/eval/state-dict tests, loads a patched COPY from a temp directory with only that one line removed. This is recorded in the results (patch_used, patch_path).

## 4. Forward-pass results (C2) — CPU

BAM (patched copy) and CBAM (pristine), constructed per-shape with matching channel count (BAM(gate_channel=C), CBAM(gate_channels=C)):

| Module | Input shape | Output shape | Shape preserved | Finite | dtype | Moved to CUDA |
|---|---|---|---|---|---|---|
| BAM | [2, 64, 32, 32] | [2, 64, 32, 32] | Yes | Yes | float32 | No |
| BAM | [2, 128, 16, 16] | [2, 128, 16, 16] | Yes | Yes | float32 | No |
| BAM | [2, 256, 8, 8] | [2, 256, 8, 8] | Yes | Yes | float32 | No |
| CBAM | [2, 64, 32, 32] | [2, 64, 32, 32] | Yes | Yes | float32 | No |
| CBAM | [2, 128, 16, 16] | [2, 128, 16, 16] | Yes | Yes | float32 | No |
| CBAM | [2, 256, 8, 8] | [2, 256, 8, 8] | Yes | Yes | float32 | No |

No NaN/Inf, no unexpected device movement, no runtime warnings captured during forward passes.

## 5. Training-mode and gradient results (C3)

| Module | Shape | loss | input grad exists | input grad finite | params with grad / total |
|---|---|---|---|---|---|
| BAM | [2, 64, 32, 32] | 0.0029 | Yes | Yes | 20 / 20 |
| BAM | [2, 128, 16, 16] | 0.0038 | Yes | Yes | 20 / 20 |
| BAM | [2, 256, 8, 8] | -0.0005 | Yes | Yes | 20 / 20 |
| CBAM | [2, 64, 32, 32] | -0.0004 | Yes | Yes | 7 / 7 |
| CBAM | [2, 128, 16, 16] | 0.0009 | Yes | Yes | 7 / 7 |
| CBAM | [2, 256, 8, 8] | 0.0008 | Yes | Yes | 7 / 7 |

loss.backward() succeeded; all trainable parameters received finite gradients.

## 6. Evaluation-mode results (C4)

Both modules, eval mode under torch.no_grad(), on all three shapes: output shape preserved and finite for every test.

## 7. State-dict results (C5)

| Module | state_dict tensors | max abs diff after save/load | equivalent |
|---|---|---|---|
| BAM | 32 | 0.0 | Yes |
| CBAM | 10 | 0.0 | Yes |

Save to temp file, reload into a fresh instance with the same constructor args, run both on the same input: outputs are numerically identical (max abs diff 0.0).

## 8. Optional compatibility results (C6)

| Module | [1,64,32,32] | [1,64,4,4] | [2,64,31,31] (odd) | torchscript trace |
|---|---|---|---|---|
| BAM | ok | ok | ok | ok |
| CBAM | ok | ok | ok | ok |

All optional checks pass, including odd spatial dimensions and TorchScript tracing.


## 8b. CBAM detailed PASS results (exact, from smoke_test_results.json)

Module: CBAM(gate_channels=C) from the pristine upstream MODELS/cbam.py (no modification). Constructed per input shape with matching channel count.

| Test | Input shape | Result | Exact values |
|---|---|---|---|
| Import (C1) | - | PASS | cbam import ok |
| Constructor | - | PASS | CBAM(gate_channels=64) ok |
| Forward (C2) | [2, 64, 32, 32] | PASS | output [2,64,32,32]; shape_preserved true; finite true; dtype float32; device_moved false; warnings [] |
| Forward (C2) | [2, 128, 16, 16] | PASS | output [2,128,16,16]; shape_preserved true; finite true; dtype float32; device_moved false |
| Forward (C2) | [2, 256, 8, 8] | PASS | output [2,256,8,8]; shape_preserved true; finite true; dtype float32; device_moved false |
| Backward (C3) | [2, 64, 32, 32] | PASS | loss -0.0003846; input_grad_exists true; input_grad_finite true; params_with_grad 7/7 |
| Backward (C3) | [2, 128, 16, 16] | PASS | loss 0.0009211; input_grad_exists true; input_grad_finite true; params_with_grad 7/7 |
| Backward (C3) | [2, 256, 8, 8] | PASS | loss 0.0007982; input_grad_exists true; input_grad_finite true; params_with_grad 7/7 |
| Eval (C4) | [2, 64, 32, 32] | PASS | output [2,64,32,32]; shape_preserved true; finite true |
| Eval (C4) | [2, 128, 16, 16] | PASS | output [2,128,16,16]; shape_preserved true; finite true |
| Eval (C4) | [2, 256, 8, 8] | PASS | output [2,256,8,8]; shape_preserved true; finite true |
| State dict (C5) | [2, 64, 32, 32] | PASS | 10 tensors; max_abs_diff 0.0; equivalent true |
| Compat (C6) | [1, 64, 32, 32] | PASS | output [1,64,32,32] |
| Compat (C6) | [1, 64, 4, 4] | PASS | output [1,64,4,4] |
| Compat (C6) | [2, 64, 31, 31] (odd) | PASS | output [2,64,31,31] |
| Compat (C6) | torchscript trace | PASS | trace ok |

**CBAM final status: PASS** (pristine upstream file, no modification, no warnings, no NaN/Inf, no CUDA movement).

## 9. Deprecated API / warnings findings

- Both bam.py and cbam.py use F.sigmoid (torch.nn.functional.sigmoid), which is deprecated in PyTorch. In torch 2.13.0+cpu it still exists and runs without emitting a warning (verified: F.sigmoid exists = True; no warnings captured in forward passes). Recommend replacing with torch.sigmoid for future cleanliness, but it is not a blocker.
- bam.py imports math but never uses it (dead import).
- bam.py references the undefined variable gate_activation (the construction bug above).
- No use of F.upsample, Variable, or other removed APIs was found.
- No CUDA-specific assumptions in either module; both are device-agnostic and never move tensors to CUDA.

## 10. Final status

- BAM: PASS (with a documented one-line patch for the upstream gate_activation bug; pristine construction fails).
- CBAM: PASS (pristine, no modification).

The upstream bug does not change the scientific design; it only requires a one-line fix (removing the dead line) when the module is integrated in Week 2.
