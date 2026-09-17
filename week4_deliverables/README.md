# Week 4 external-hardware package

This package contains the unchanged Week 4 configuration and runner for the
full one-seed experiment: seed 0, 100 epochs, full CIFAR-10 training split,
full 10,000-image clean test evaluation, and full 10,000-image-per-severity
evaluation for brightness, contrast, defocus_blur, and elastic_transform.

Install dependencies from `requirements-week4.txt`, place CIFAR-10 and the
four full CIFAR-10-C arrays where the existing `config.py` expects them, then
run:

    python scripts/run_week4_full.py

The included JSON results and report are the existing blocked-status records.
No Week 4 training was run in the constrained environment. No three-seed run
was started. The package excludes local data arrays and all Week 3 checkpoints
and logs.
