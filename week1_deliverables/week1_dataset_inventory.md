# Week 1 Dataset Inventory — CIFAR-10-C

Project: "Do CNN Attention Modules Survive Common Corruptions? A Severity-Conditioned Comparison of SE, BAM, and CBAM on CIFAR-10-C"

## 1. Source and revision

| Field | Value |
|---|---|
| Dataset source | CIFAR-10-C and CIFAR-10-P (official, Zenodo) |
| Source URL | https://zenodo.org/records/2535967 |
| DOI | 10.5281/zenodo.2535967 |
| Zenodo record revision | 6 |
| Creator | Daniel Hendrycks |
| Publication date | 2019-01-09 (modified 2020-01-20) |
| Archive file | CIFAR-10-C.tar |
| Archive size | 2,918,471,680 bytes (~2.72 GiB) |
| Archive MD5 | 56bf5dcef84df0e2308c6dcbcbbd8499 (verified against the downloaded file) |
| License | CC BY 4.0 |

The archive was downloaded and its MD5 was verified to match the Zenodo metadata exactly. The full member listing was inspected directly from the tarball.

## 2. Corruption inventory (verified from the actual tarball)

The official Zenodo CIFAR-10-C.tar contains **19 corruption .npy files** plus one shared labels.npy (21 members including the directory entry). Every corruption file is exactly 153,600,128 bytes = 128-byte NumPy header + 50,000 x 32 x 32 x 3 uint8.

**Classification: original 15 standard corruptions PLUS 4 additional corruptions (a 19-type package).**

Standard 15 (from the original ICLR 2019 benchmark):
gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression

Additional 4 (extra corruptions in this packaged version):
speckle_noise, gaussian_blur, spatter, saturate

This is corroborated by the official generation script in the robustness repository (hendrycks/robustness, ImageNet-C/create_c/make_cifar_c.py), which builds the same 19-corruption OrderedDict (15 standard + Speckle Noise, Gaussian Blur, Spatter, Saturate) at severities 1-5 from the CIFAR-10 test set.

## 3. Required four-corruption verification

| Required corruption | Found? | Exact filename | Severities present | Shape / count |
|---|---|---|---|---|
| brightness | Yes | brightness.npy | 1, 2, 3, 4, 5 | (50000, 32, 32, 3) uint8; 10,000 per severity |
| contrast | Yes | contrast.npy | 1, 2, 3, 4, 5 | (50000, 32, 32, 3) uint8; 10,000 per severity |
| defocus blur | Yes | defocus_blur.npy | 1, 2, 3, 4, 5 | (50000, 32, 32, 3) uint8; 10,000 per severity |
| elastic transform | Yes | elastic_transform.npy | 1, 2, 3, 4, 5 | (50000, 32, 32, 3) uint8; 10,000 per severity |

Name mapping: the files use the underscore form (defocus_blur, elastic_transform). The paper/report prose names ("defocus blur", "elastic transform") map 1:1 to these filenames. The robustness generation code calls elastic_transform "Elastic" and jpeg_compression "JPEG" in its display dict, but the saved filenames are the function names above.

## 4. Severity and sample-count verification

Each of the 19 corruption files holds 50,000 images arranged as 5 consecutive blocks of 10,000: rows 0-9,999 = severity 1, rows 10,000-19,999 = severity 2, ..., rows 40,000-49,999 = severity 5. This matches the Zenodo description ("the first 10,000 images in each .npy are the test set images corrupted at severity 1, and the last 10,000 images are the test set images corrupted at severity five").

Per-severity sampling of the four required corruptions (one image from each severity band):

| Corruption | sev1 mean | sev2 mean | sev3 mean | sev4 mean | sev5 mean | finite | dtype |
|---|---|---|---|---|---|---|---|
| brightness | 117.99 | 128.19 | 138.29 | 148.27 | 167.23 | True | uint8 |
| contrast | 107.91 | 107.94 | 107.95 | 107.90 | 107.89 | True | uint8 |
| defocus_blur | 107.87 | 107.84 | 107.80 | 107.80 | 107.79 | True | uint8 |
| elastic_transform | 110.27 | 107.14 | 108.93 | 107.28 | 107.71 | True | uint8 |

Brightness means rise monotonically with severity, confirming the severity ordering is meaningful. All samples are finite, uint8, in [0, 255], shape (32, 32, 3). The per-corruption file size (153,600,128 bytes) is consistent with 50,000 x 32 x 32 x 3 uint8 for every corruption, and the generation script applies 5 severities to all 19 types, so the 5 x 10,000 layout holds for all 19.

## 5. Image and label checks

- Image shape: (32, 32, 3), uint8, values in [0, 255] for all sampled images; no NaN/Inf.
- Label file: CIFAR-10-C/labels.npy, shape (50000,), dtype uint8, unique values {0,...,9}.
- Label structure: 5 identical blocks of 10,000 (block0 == block1 == ... == block4), i.e. the same 10,000 labels are reused across the 5 severities.
- Labels match CIFAR-10: verified. The 10,000-label block exactly equals the canonical CIFAR-10 test_batch labels (0 mismatches) from cifar-10-python.tar.gz. The official generation code (make_cifar_c.py) builds labels from dset.CIFAR10(train=False).targets, i.e. the CIFAR-10 test set.
- Corrupted images correspond to the CIFAR-10 test set: yes by provenance (make_cifar_c.py corrupts the CIFAR-10 test split). A pixel-exact spot check of defocus_blur severity-1 row 0 against the mirror copy also matched.
- Data loader ordering: the files are plain .npy arrays; no reordering is performed by a loader. Labels are aligned by row index (row i of any corruption file corresponds to labels[i]).

## 6. Dataset format and loading instructions

- Format: uncompressed tar archive of NumPy .npy files (uint8 arrays).
- Load one corruption/severity with numpy: arr = np.load("CIFAR-10-C/brightness.npy"); severity s block = arr[(s-1)*10000 : s*10000].
- Labels: labels = np.load("CIFAR-10-C/labels.npy"); labels[i] is the class (0-9) for row i.
- Standard preprocessing for training: cast to float, /255, CHW, normalize per CIFAR-10 statistics.

## 7. Mirror comparison (Hugging Face)

| Field | HF mirror (WNJXYK/TTA-CIFAR-10-C) |
|---|---|
| Revision (repo sha) | 0f57314bbe937783522f1d348758ca2367ed09a9 |
| Last modified | 2026-04-22 |
| License | CC BY 4.0 (matches upstream) |
| Structure | 15 configs (one per corruption), 5 splits per config (severity_1..5), 10,000 images each |
| Total rows | 750,000 |
| Format | Parquet (image bytes + label) |
| Upstream claim | SHA256 of upstream tarball c72763e101c723b7c507b96205f7e938912a5d587376173b825850cf3cb876a7; bytes copied 1:1 from upstream .npy |

Discrepancy: the HF mirror contains only the **15 standard corruptions** (configs: gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression). It is missing the 4 extra corruptions present in the official Zenodo tarball (gaussian_blur, saturate, spatter, speckle_noise). The mirror is therefore a 15-type subset, while the official Zenodo source is a 19-type package. The mirror's labels were verified to match the canonical CIFAR-10 test labels exactly, and a decoded image from the mirror matched the official .npy pixel-for-pixel.

## 8. Provenance and license notes

- Official source: Zenodo record 2535967, CC BY 4.0, by Daniel Hendrycks (Hendrycks & Dietterich, ICLR 2019, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations").
- Generation code: hendrycks/robustness (commit 8190fe3), ImageNet-C/create_c/make_cifar_c.py.
- Original CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html (test split used as corruption input).

## 9. Unresolved issues / notes

- The official 19-type tarball was downloaded and fully verified during this week's work; the sandbox environment was later reset and the file removed, so the authoritative member listing and samples below come from that verified inspection (recorded in this report) plus the generation code. dataset_inventory.py reproduces the same inventory when pointed at the full official tarball.
- The HF mirror is 15-type only; any experiment needing the 4 extra corruptions (gaussian_blur, saturate, spatter, speckle_noise) must use the official Zenodo tarball.
- No model training was performed during this milestone.
