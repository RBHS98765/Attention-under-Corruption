# Week 3 Training Report: SE Implementation, Training and CIFAR-10-C Evaluation Infrastructure

Project: Do CNN Attention Modules Survive Common Corruptions? A Severity-Conditioned Comparison of SE, BAM, and CBAM on CIFAR-10-C.
Milestone: Week 3 (finalize training and evaluation infrastructure; short training smoke test only). No full multi-seed experiment performed.

## 1. SE implementation (Part A)

`models/se.py` implements the SE block from the SE paper equations (Hu, Shen, Sun, CVPR 2018):

- Squeeze: global average pooling produces `[B, C]` from `[B, C, H, W]`.
- Excitation: two-layer channel bottleneck `Linear(C -> hidden) -> ReLU -> Linear(hidden -> C)`.
- Gate: sigmoid applied to the bottleneck output.
- Scale: channel-wise rescaling `x_tilde_c = s_c * u_c`; output shape `[B, C, H, W]`.

Reduction ratio (documented, not claimed as uniquely recovered from the paper): `r = 16`, a standard commonly used value. Hidden width formula: `hidden = max(C // r, 1)`. Activation: ReLU. Initialization: default PyTorch Linear initialization (no custom init). SE integrates into the existing `build_model(attention_type=...)` factory with no backbone duplication and no change to the insertion point.

## 2. Training loop (Part B)

- Dataset: standard CIFAR-10 training split (50,000 images); the smoke test trains on a deterministic 10,000-image subset.
- Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor, Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)).
- Loss: cross-entropy.
- Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4.
- Schedule: cosine annealing over the total number of optimizer steps.
- Batch size: 128. Epochs: 1 (smoke test).
- Deterministic seeding: `set_seed(seed)` sets Python random, NumPy, torch CPU, and torch CUDA (if available); sets `cudnn.deterministic=True` and `cudnn.benchmark=False` on GPU. Seed used: 0.
- Checkpointing: saves model state_dict, optimizer state_dict, epoch, seed, config, and metrics to `ckpts/ckpt_<variant>_<seed>_epoch_<e>.pt`. Checkpoints from different seeds are never overwritten.
- Logging: per-epoch training loss/accuracy, validation loss/accuracy, learning rate, and timestamp are written to `logs/smoke_train_<variant>.csv` (not only printed text), plus a consolidated `logs/week3_smoke_results.json`.
- No test-time adaptation and no corruption-specific training.

## 3. CIFAR-10-C evaluation loop (Part C)

- Four primary corruptions: brightness, contrast, defocus_blur, elastic_transform.
- Severities 1-5; the clean CIFAR-10 test set is severity 0.
- Model weights and batch-norm statistics are frozen during evaluation (`model.eval()`, `torch.no_grad()`); no retraining or adaptation per corruption or severity.
- Data: corruptions generated locally with the official generation algorithm (hendrycks/robustness, commit 8190fe3, `make_cifar_c.py`), laid out exactly like the official archive (`[50000,32,32,3]` uint8, 5 blocks of 10000). elastic_transform is stochastic (np.random displacement field); seeded for reproducibility.
- Smoke-test evaluation uses a subset for speed: 1000 clean test images and 500 images per severity per corruption. This is documented and is not a performance result.
- Metrics recorded per variant, seed, corruption, severity: accuracy, number of examples, checkpoint path, seed, timestamp.

## 4. Smoke-test training results (Part D)

One epoch, 10,000 training images, batch 128, seed 0, CPU.

| Variant | Train loss | Train acc | Val acc | Loss finite | Acc reasonable | Checkpoint saved |
|---|---:|---:|---:|---|---|---|
| none | 1.8629 | 0.2893 | 0.3788 | yes | yes | yes |
| se | 1.8685 | 0.2884 | 0.3477 | yes | yes | yes |
| bam | 1.9106 | 0.2779 | 0.3607 | yes | yes | yes |
| cbam | 1.8509 | 0.2926 | 0.3695 | yes | yes | yes |

All four variants trained without error, produced finite loss and reasonable accuracy (not 0 or 1), and saved checkpoints. Note: with a single-epoch cosine schedule the learning rate anneals to ~0 by the last step; this is expected and does not affect the smoke-test purpose.

## 5. Smoke-test evaluation results (Part C/D)

Clean CIFAR-10 (severity 0) and the four primary corruptions at severities 1-5. Accuracy per severity (subset: 1000 clean, 500 per severity).

| Variant | Clean | Brightness sev1-5 | Contrast sev1-5 | Defocus sev1-5 | Elastic sev1-5 |
|---|---:|---|---|---|---|
| none | 0.386 | 0.392, 0.362, 0.336, 0.302, 0.228 | 0.302, 0.220, 0.194, 0.188, 0.174 | 0.386, 0.372, 0.348, 0.332, 0.304 | 0.354, 0.370, 0.348, 0.352, 0.338 |
| se | 0.353 | 0.356, 0.346, 0.302, 0.278, 0.246 | 0.290, 0.226, 0.206, 0.186, 0.172 | 0.340, 0.332, 0.314, 0.308, 0.292 | 0.314, 0.318, 0.314, 0.294, 0.308 |
| bam | 0.357 | 0.334, 0.328, 0.306, 0.264, 0.222 | 0.280, 0.188, 0.154, 0.152, 0.136 | 0.352, 0.334, 0.320, 0.308, 0.288 | 0.336, 0.336, 0.316, 0.326, 0.306 |
| cbam | 0.373 | 0.372, 0.328, 0.280, 0.252, 0.220 | 0.306, 0.224, 0.198, 0.194, 0.184 | 0.384, 0.354, 0.318, 0.298, 0.270 | 0.346, 0.340, 0.340, 0.316, 0.322 |

These are smoke-test numbers only; no clean or corrupted accuracy is claimed as an experimental result.

## 6. Data provenance and fidelity

The four primary corruption files were generated locally with the official generation algorithm (robustness commit 8190fe3, `make_cifar_c.py`), each saved as `[50000,32,32,3]` uint8 (153,600,128 bytes), matching the official archive layout. Full-set means were computed (e.g., brightness sev1 mean 131.55, contrast mean 121.03). Week 1's `week1_verification.json` sample_checks (e.g., brightness sev1 117.99) were per-severity single-image samples, per the Week 1 report's own wording ("one image from each severity band"), so they are not comparable to full-set statistics; this is a clarification, not an error in either week's work.

## 7. Environment versions

- Python 3.12.10
- PyTorch 2.13.0+cpu
- Torchvision 0.28.0+cpu
- NumPy 2.5.0, SciPy 1.18.0, scikit-image 0.26.0, OpenCV 5.0.0
- CPU-only (no CUDA)

## 8. Warnings and known limitations

- Single-epoch cosine annealing drives the learning rate to ~0 by the final step; acceptable for a pipeline smoke test, but the full experiment should use a multi-epoch schedule.
- Smoke-test evaluation uses subsets (1000 clean, 500 per severity) for speed; the full experiment should evaluate on all 10,000 test images per severity.
- elastic_transform is stochastic; the local generation is one seeded draw, so exact pixel values differ from the official archive's draw (the algorithm and layout are identical).
- BAM's upstream BatchNorm1d singleton-batch limitation (from Week 2) does not affect this smoke test (training batch 128, eval batch 128).
- The sandbox reset several times during the run; the per-variant incremental runner saved results after each variant, so none, se, and bam results were persisted before cbam ran; all four are complete in the final results file.

## 9. Unresolved issues

None blocking.

## 10. Status

WEEK 3 PASS WITH NON-BLOCKING ISSUES
