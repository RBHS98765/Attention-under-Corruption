# Week 1 Environment Report

Project: "Do CNN Attention Modules Survive Common Corruptions? A Severity-Conditioned Comparison of SE, BAM, and CBAM on CIFAR-10-C"

## 1. Software environment

| Component | Version |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.13.0+cpu |
| Torchvision | 0.28.0+cpu |
| NumPy | (installed) |
| PyArrow | 25.0.1 (installed for parquet inspection) |
| pip | 26.1.2 |

## 2. Hardware / OS

| Component | Value |
|---|---|
| OS | Linux (kernel 6.8.0-86-generic, x86_64, glibc 2.36) |
| CPU | 48 cores |
| RAM | 377 GB |
| GPU | None (CPU-only build; CUDA not available in this sandbox) |
| CUDA | Not available (torch.version.cuda = None) |

Note: the smoke tests ran on CPU. GPU was not available, so the GPU-only optional test was not executed; this is not a blocker because the modules are device-agnostic and the CPU forward/backward/state-dict tests all passed.

## 3. Repository commits

| Repository | URL | Commit hash | Last commit date |
|---|---|---|---|
| attention-module (BAM/CBAM) | https://github.com/Jongchan/attention-module | 459efad0e05ee7dde50c41ca10a3d0800bc3792a | 2022-09-12 |
| robustness (Hendrycks) | https://github.com/hendrycks/robustness | 8190fe329f5f072a06e3a2aea02bb5dda69aed9f | 2022-08-24 |
| SENet (SE reference) | https://github.com/hujie-frank/SENet | 0262d43d44c561fd53c3dba210cc8bacfc60500d | 2019-02-25 |

## 4. Dataset revision / source version

| Item | Value |
|---|---|
| CIFAR-10-C official | Zenodo record 2535967, revision 6, DOI 10.5281/zenodo.2535967 |
| CIFAR-10-C.tar | 2,918,471,680 bytes, MD5 56bf5dcef84df0e2308c6dcbcbbd8499 |
| HF mirror | WNJXYK/TTA-CIFAR-10-C, repo sha 0f57314bbe937783522f1d348758ca2367ed09a9, last modified 2026-04-22 |
| CIFAR-10 (labels cross-check) | cifar-10-python.tar.gz (canonical test_batch) |

## 5. Exact commands used

Environment install:
- python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
- python3 -m pip install pyarrow

Repository clones:
- git clone --depth 1 https://github.com/Jongchan/attention-module.git
- git clone --depth 1 https://github.com/hendrycks/robustness.git
- git clone --depth 1 https://github.com/hujie-frank/SENet.git

Dataset:
- curl -o CIFAR-10-C.tar 'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1'  (2,918,471,680 bytes; MD5 verified)
- curl -o cifar-10-python.tar.gz 'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz' (canonical CIFAR-10 for label check)

Smoke tests:
- python3 smoke_test_attention_modules.py  (output also saved to smoke_test_results.json and smoke_test.log)

Inventory:
- python3 dataset_inventory.py CIFAR-10-C.tar --cifar10 cifar-10-python.tar.gz --output week1_dataset_inventory.json
  (validated on a 9-corruption subset tarball because the full official tarball was removed by a sandbox reset; the script is designed for the full official tarball)

Inspection:
- tar member listing and .npy sampling via Python tarfile + numpy (see week1_dataset_inventory.md)
- HF mirror parquet inspected via the datasets-server API and pyarrow
