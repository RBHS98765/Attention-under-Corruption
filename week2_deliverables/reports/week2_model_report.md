# Week 2 Model Report: Matched Backbone with Pluggable Attention (SE, BAM, CBAM)

Project: Do CNN Attention Modules Survive Common Corruptions? A Severity-Conditioned Comparison of SE, BAM, and CBAM on CIFAR-10-C.
Milestone: Week 2 (verification of the matched backbone and the pluggable attention slot). No training performed.

## 1. Architecture

`MatchedBackbone` is a small residual CNN designed for CIFAR-10 `[B, 3, 32, 32] -> [B, 10]`. It avoids ImageNet-specific assumptions: no 7x7 initial convolution, no initial stride 2, no ImageNet-sized pooling. The exact same convolutional structure is used in every variant; only the attention slot differs.

```
Input [B, 3, 32, 32]
  Stem:  Conv2d(3, 64, 3, stride=1, padding=1) -> BatchNorm2d -> ReLU          (32x32)
  Stage 1:  ResidualBlock(64,64, stride=1) x2                                 (32x32)
  Stage 2:  ResidualBlock(64,128, stride=2) x1                                  (16x16)
  Stage 3:  ResidualBlock(128,256, stride=2) x1                                 (8x8)
  Head:   AdaptiveAvgPool2d(1) -> flatten -> Linear(256, 10)
Output [B, 10]
```

- Number of residual blocks: 4 (2 + 1 + 1).
- Channel widths per block: 64, 64, 128, 256.
- Downsampling: stage 2 and stage 3 use stride 2 on the first conv; the shortcut uses a 1x1 stride-2 conv.
- Normalization: BatchNorm2d after every conv (stem and residual convs).
- Activation: ReLU.
- Residual connections: standard identity/shortcut addition in every block.

## 2. Exact attention insertion point

Attention is applied **after the residual block's convolutional feature transformation (conv2 + bn2) and before the residual addition**, in every residual block. This exact location is used identically for SE, BAM, and CBAM. Each block's attention module is configured with that block's own channel count (64, 64, 128, 256). The original BAM bottleneck placement is deliberately not used; Week 2 tests a matched insertion condition.

## 3. Constructor details

- Factory: `build_model(attention_type=None|"se"|"bam"|"cbam", num_classes=10, se_reduction=16, bam_reduction=16, cbam_reduction=16)`, built once per variant from the shared `MatchedBackbone` class (no backbone duplication).
- Attention factory: `build_attention(attention_type, channels, ...)` returns `Identity()` for `none`, `SEBlock(channels, reduction=16)`, `BAM(channels, reduction_ratio=16, dilation_conv_num=2, dilation_val=4)`, or `CBAM(channels, reduction_ratio=16, pool_types=['avg','max'], no_spatial=False)`.
- SE: `hidden = max(C // r, 1)`; global average pool -> Linear(C, hidden) -> ReLU -> Linear(hidden, C) -> Sigmoid -> channel rescale.
- BAM/CBAM: copied from the Week 1 verified upstream `attention-module` (commit `459efad0`), BAM with the one-line patch applied (see section 8).

## 4. Parameter table

| Variant | Backbone params | Attention params | Classifier params | Total trainable params | MACs (conv+linear) |
|---|---:|---:|---:|---:|---:|
| Plain CNN | 1,301,578 | 0 | 2,570 | 1,301,578 | 270,207,488 |
| SE | 1,301,578 | 11,808 | 2,570 | 1,313,386 | 270,218,752 |
| BAM | 1,301,578 | 24,164 | 2,570 | 1,325,742 | 272,458,240 |
| CBAM | 1,301,578 | 12,208 | 2,570 | 1,313,786 | 270,462,080 |

- Backbone parameters (total minus attention) are **exactly identical** across all four variants: 1,301,578.
- Non-trainable parameters: 0 for every variant (nothing frozen).
- Total-parameter differences are attributable only to the attention blocks: SE adds 11,808, BAM adds 24,164, CBAM adds 12,208, each equal to the variant's attention-only parameter count.
- Attention blocks per variant: 4 (one per residual block).
- Per-block attention channel counts verified: 64, 64, 128, 256.

## 5. Test matrix and results

`scripts/smoke_test_week2.py` runs the 11 required tests (Part D) for every variant on inputs `[2,3,32,32]`, `[1,3,32,32]`, `[4,3,32,32]`, plus Part E attention sanity and Part F one-batch training. Exit code 0; all checks pass.

| Test | none | se | bam | cbam |
|---|---|---|---|---|
| 1. Construction | PASS | PASS | PASS | PASS |
| 2. Forward pass | PASS | PASS | PASS | PASS |
| 3. Output shape [B,10] | PASS | PASS | PASS | PASS |
| 4. Finite output | PASS | PASS | PASS | PASS |
| 5. Training mode | PASS | PASS | PASS | PASS |
| 6. Backward pass | PASS | PASS | PASS | PASS |
| 7. Finite gradient | PASS | PASS | PASS | PASS |
| 8. Evaluation mode | PASS | PASS | PASS | PASS |
| 9. State-dict save/load | PASS | PASS | PASS | PASS |
| 10. CPU device | PASS | PASS | PASS | PASS |
| 11. Parameter accounting | PASS | PASS | PASS | PASS |

Attention sanity (Part E): attention outputs preserve feature-map shape, no NaN/Inf, gradients reach all attention parameters, attention parameters are included in `model.parameters()`, and the identity baseline contains zero attention parameters. All PASS.

One-batch training smoke test (Part F), random mini-batch B=8, cross-entropy loss, one SGD step:

| Variant | Loss | Loss finite | Params changed |
|---|---:|---|---|
| none | 2.5165 | yes | yes |
| se | 2.2418 | yes | yes |
| bam | 2.3767 | yes | yes |
| cbam | 2.3995 | yes | yes |

These are integration checks only; no accuracy is reported as an experimental result.

## 6. Forward-pass results

All four variants return `[B, 10]` for B in {1, 2, 4}, all logits finite, float32, CPU. Default `logits = model(x)` is compatible with a standard PyTorch training loop; `logits, attention_info = model(x, return_attention=True)` is available as a debug path without changing the default output.

## 7. Backward and state-dict results

Backward: losses backpropagate cleanly; every trainable parameter (including all attention parameters) receives a finite gradient. State dict: `load_state_dict` round-trip reproduces identical outputs (`torch.allclose` true) for every variant.

## 8. BAM patch explanation

- Upstream repository: https://github.com/Jongchan/attention-module, commit `459efad0e05ee7dde50c41ca10a3d0800bc3792a`, MIT license.
- The upstream `MODELS/bam.py` contains one dead, undefined assignment `self.gate_activation = gate_activation` in `ChannelGate.__init__` that raises `NameError` at construction (confirmed in Week 1).
- Patch: `bam_gate_activation_fix.patch` removes exactly that one line. AST comparison of the project copy against upstream confirms every `forward` method and the `ChannelGate`/`SpatialGate` constructors are identical; the only additional difference is an optional reduction/dilation signature extension on `BAM.__init__` whose defaults match upstream. Forward computation is unchanged.
- The upstream repository is never modified (verified `git status` clean at `459efad`). The patched module lives only in the project copy.

## 9. Environment versions

- Python 3.12.10
- PyTorch 2.13.0+cpu
- Torchvision 0.28.0+cpu
- CPU-only build (no CUDA), Linux x86_64

## 10. Known warnings

- Upstream BAM's `ChannelGate` uses `BatchNorm1d` after global pooling; a singleton B=1 training-mode forward raises `ValueError` because batch statistics cannot be estimated. Shape probes therefore use eval mode for B=1, while the optimizer smoke test uses B=8 and passes. This is an upstream property, not a project defect.
- `BAM.__init__` signature extended with optional `reduction_ratio`/`dilation` arguments (default values match upstream); forward computation unchanged.

## 11. Unresolved issues

None blocking. The BAM singleton-batch limitation is the only known caveat and does not affect the required training-mode smoke test or evaluation.

## 12. Status

WEEK 2 PASS WITH NON-BLOCKING ISSUES
