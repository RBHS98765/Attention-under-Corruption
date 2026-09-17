# Week 4 Training Report: Full One-Seed Pipeline Validation

Project: Do CNN Attention Modules Survive Common Corruptions? A Severity-Conditioned Comparison of SE, BAM, and CBAM on CIFAR-10-C.

## 1. Scope

Week 4 was configured for one seed only, seed 0. The three-seed experiment was not started. The frozen Week 3 architecture, attention insertion point, SE reduction ratio, BAM patch, CBAM settings, optimizer family, augmentation, full CIFAR-10 training split, and full CIFAR-10-C evaluation design were preserved.

## 2. Frozen full-run configuration

- Variants: Plain CNN, SE, BAM, CBAM.
- Training data: full CIFAR-10 training split, 50,000 images.
- Schedule: 100 epochs, cosine annealing over the full schedule.
- Optimizer: SGD, learning rate 0.1, momentum 0.9, weight decay 5e-4.
- Batch size: 128.
- Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip, CIFAR-10 normalization.
- Seed: 0, with Python, NumPy, PyTorch CPU/CUDA seeding and deterministic cuDNN settings where applicable.
- Clean evaluation: all 10,000 CIFAR-10 test images.
- Corruption evaluation: brightness, contrast, defocus_blur, elastic_transform, severities 1-5, all 10,000 images per corruption and severity.
- Evaluation: `model.eval()` and `torch.no_grad()`, with model weights and batch-normalization statistics frozen.

The runnable configuration is `week4_config.json`. The runnable full-run entry point is `scripts/run_week4_full.py`.

## 3. Diagnostic result and blocker

The execution environment has 64 visible CPUs but is cgroup-limited to two effective CPU cores (`cpu.max = 200000/100000`) and has no CUDA device. Timing the frozen Week 2/3 model on the full architecture produced approximately:

- Plain CNN, batch 128: 1.802 seconds per training step at two threads.
- Approximately 390 steps per full 50,000-image epoch.
- Approximately 11.7 minutes per variant per epoch.
- Approximately 78 CPU-hours for four variants at the required 100-epoch schedule, before the full clean and CIFAR-10-C evaluation.

A shorter benchmark also confirmed the bottleneck is model computation rather than data augmentation. Reducing the schedule or training subset would violate the Week 4 requirements, so I did not silently substitute a smaller experiment.

## 4. What was and was not run

The full one-seed Week 4 training run was **not completed** because the required schedule is not feasible within the current CPU-limited execution environment. The full three-seed experiment was not started.

No Week 4 clean or corrupted accuracy claims are made. The Week 3 one-epoch smoke results remain pipeline smoke results only and are not reused as Week 4 results.

The blocker is **purely compute availability, not a model or pipeline defect**. The model construction, training-loop integration, checkpointing, and evaluation path were verified by the Week 3 smoke tests for Plain CNN, SE, BAM, and CBAM; Week 3 concluded `WEEK 3 PASS WITH NON-BLOCKING ISSUES`.

The full-run runner is ready to execute on a machine with adequate CPU/GPU capacity. It writes:

- `ckpts/ckpt_<variant>_seed0_epoch_<e>.pt`
- `logs/train_<variant>_seed0.csv`
- `logs/eval_<variant>_seed0_clean.csv`
- `logs/eval_<variant>_seed0_corruption.csv`
- `results/week4_clean_results.json`
- `results/week4_corruption_results.json`

## 5. Known limitations

- Week 4 acceptance criteria requiring completed 100-epoch training, reasonable clean accuracy, and full corruption accuracy cannot be assessed until the full one-seed run executes on adequate compute.
- The local primary-corruption arrays are available and have the required full layout: 10,000 images per severity for each of the four primary corruptions. They are not copied into the deliverables archive because they are approximately 600 MB.
- No accuracy, robustness ranking, or scientific performance conclusion is reported for Week 4.

## 6. Status

WEEK 4 BLOCKED
