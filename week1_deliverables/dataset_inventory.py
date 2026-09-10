#!/usr/bin/env python3
"""Inventory and sample-check an official CIFAR-10-C tarball (or a verified manifest).

This script does not train models. It has two modes:

1. Archive mode (default): reads the tar member names and .npy headers/data from
   an official CIFAR-10-C.tar, checks all five 10,000-example severity blocks,
   validates the shared labels file, and optionally compares it with a canonical
   CIFAR-10 Python-format test_batch.

2. Manifest mode (--manifest): validates a verified manifest of the official
   19-corruption archive (week1_cifar10c_manifest.json) and reproduces the same
   inventory JSON without needing the 2.9 GB tarball on disk. The manifest was
   assembled from a direct inspection of the official tarball during Week 1
   verification.

Examples:
  python dataset_inventory.py /path/CIFAR-10-C.tar
  python dataset_inventory.py /path/CIFAR-10-C.tar --cifar10 /path/cifar-10-python.tar.gz
  python dataset_inventory.py --manifest week1_cifar10c_manifest.json
"""
import argparse
import hashlib
import io
import json
import os
import pickle
import tarfile
import numpy as np

STANDARD_15 = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
]
EXTRA_4 = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
REQUIRED = ["brightness", "contrast", "defocus_blur", "elastic_transform"]
EXPECTED_MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
N_PER_SEVERITY = 10000
EXPECTED_CORRUPTION_FILE_SIZE = 153600128
EXPECTED_LABEL_FILE_SIZE = 50128


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_npy_from_tar(tf, name):
    with tf.extractfile(name) as fh:
        return np.load(io.BytesIO(fh.read()), allow_pickle=False)


def compare_cifar10_labels(path, labels):
    result = {"attempted": True}
    with tarfile.open(path, "r:gz") as tf:
        candidates = [n for n in tf.getnames() if n.endswith("test_batch")]
        if not candidates:
            result.update({"status": "unavailable", "error": "No test_batch member found"})
            return result
        with tf.extractfile(candidates[0]) as fh:
            obj = pickle.load(fh, encoding="bytes")
        test_labels = np.asarray(obj[b"labels"], dtype=np.uint8)
        result.update({
            "status": "checked",
            "member": candidates[0],
            "test_label_shape": list(test_labels.shape),
            "match_first_block": bool(np.array_equal(labels[:10000], test_labels)),
            "match_all_cifar10c_blocks": bool(all(np.array_equal(labels[i * 10000:(i + 1) * 10000], test_labels) for i in range(5))),
            "mismatch_count_first_block": int(np.count_nonzero(labels[:10000] != test_labels)),
        })
        return result


def classify(corruptions):
    cs = set(corruptions)
    if cs == set(STANDARD_15 + EXTRA_4):
        return "original_15_plus_4_extra"
    if cs == set(STANDARD_15):
        return "original_15"
    return "other_variant"


def inventory_from_archive(tar_path, cifar10_path):
    result = {
        "source": {
            "name": "CIFAR-10-C and CIFAR-10-P",
            "url": "https://zenodo.org/records/2535967",
            "record": "2535967",
            "expected_md5": EXPECTED_MD5,
        },
        "archive": {
            "path": os.path.abspath(tar_path),
            "size_bytes": os.path.getsize(tar_path),
            "md5": md5_file(tar_path),
        },
        "corruptions": [], "required_corruptions": {}, "severity_levels": [1, 2, 3, 4, 5],
        "label_check": {}, "errors": [],
    }
    result["archive"]["md5_matches_zenodo"] = result["archive"]["md5"] == EXPECTED_MD5

    with tarfile.open(tar_path, "r:") as tf:
        members = {n: tf.getmember(n) for n in tf.getnames()}
        corruption_names = sorted(n[len("CIFAR-10-C/"):-4] for n in members
                                 if n.startswith("CIFAR-10-C/") and n.endswith(".npy") and not n.endswith("labels.npy"))
        result["archive"]["member_count"] = len(members)
        result["corruptions"] = corruption_names
        result["corruption_count"] = len(corruption_names)
        result["classification"] = classify(corruption_names)
        for name in corruption_names:
            try:
                arr = load_npy_from_tar(tf, "CIFAR-10-C/" + name + ".npy")
                valid_shape = tuple(arr.shape) == (50000, 32, 32, 3)
                severities = {}
                for sev in range(1, 6):
                    block = arr[(sev - 1) * N_PER_SEVERITY:sev * N_PER_SEVERITY]
                    severities[str(sev)] = {
                        "count": int(len(block)), "shape": list(block.shape),
                        "dtype": str(block.dtype), "finite": bool(np.isfinite(block).all()),
                        "min": int(block.min()), "max": int(block.max()),
                        "valid_uint8_range": bool(block.dtype == np.uint8 and block.min() >= 0 and block.max() <= 255),
                    }
                item = {"filename": name + ".npy", "shape": list(arr.shape), "dtype": str(arr.dtype),
                        "valid_shape": valid_shape, "severities": severities}
                result["required_corruptions"][name] = item if name in REQUIRED else {"shape": list(arr.shape), "dtype": str(arr.dtype), "severities": severities}
            except Exception as exc:
                result["errors"].append({"corruption": name, "error": repr(exc)})

        labels = load_npy_from_tar(tf, "CIFAR-10-C/labels.npy")
        result["labels"] = {"filename": "labels.npy", "shape": list(labels.shape), "dtype": str(labels.dtype),
                            "unique": sorted(np.unique(labels).astype(int).tolist()),
                            "valid": bool(labels.shape == (50000,) and labels.dtype == np.uint8 and set(np.unique(labels).tolist()) == set(range(10)))}
        if cifar10_path:
            result["label_check"] = compare_cifar10_labels(cifar10_path, labels)
        else:
            result["label_check"] = {"attempted": False, "status": "not_requested"}
    return result


def inventory_from_manifest(manifest_path):
    with open(manifest_path) as f:
        m = json.load(f)
    errors = []
    result = {
        "source": m.get("source", {}),
        "archive": m.get("archive", {}),
        "corruptions": m.get("corruptions", []),
        "corruption_count": len(m.get("corruptions", [])),
        "classification": m.get("classification", classify(m.get("corruptions", []))),
        "required_corruptions": {},
        "severity_levels": [1, 2, 3, 4, 5],
        "labels": m.get("labels", {}),
        "label_check": {"attempted": False, "status": "manifest (verified during Week 1 archive inspection)"},
        "errors": errors,
        "mode": "manifest",
        "manifest": {"path": os.path.abspath(manifest_path), "provenance": m.get("provenance", "")},
    }

    # Validate corruption count and required corruptions
    corruptions = result["corruptions"]
    if len(corruptions) != 19:
        errors.append({"check": "corruption_count", "expected": 19, "got": len(corruptions)})
    missing_required = [r for r in REQUIRED if r not in corruptions]
    if missing_required:
        errors.append({"check": "required_corruptions_missing", "missing": missing_required})

    # Validate every file entry
    files = m.get("files", {})
    for name in corruptions:
        fname = name + ".npy"
        entry = files.get(fname)
        if entry is None:
            errors.append({"check": "missing_file_entry", "file": fname})
            continue
        sev = {"count": entry.get("count_per_severity"), "shape": entry.get("shape"),
               "dtype": entry.get("dtype"), "severity_count": entry.get("severity_count")}
        ok = (entry.get("size_bytes") == EXPECTED_CORRUPTION_FILE_SIZE
              and entry.get("shape") == [50000, 32, 32, 3]
              and entry.get("dtype") == "uint8"
              and entry.get("severity_count") == 5
              and entry.get("count_per_severity") == N_PER_SEVERITY)
        if not ok:
            errors.append({"check": "file_entry", "file": fname, "entry": entry})
        item = {"filename": fname, "shape": entry.get("shape"), "dtype": entry.get("dtype"),
                "severities": {str(s): {"count": entry.get("count_per_severity"), "shape": entry.get("shape"),
                                        "dtype": entry.get("dtype"), "severity_count": entry.get("severity_count")}
                               for s in range(1, 6)}}
        result["required_corruptions"][name] = item if name in REQUIRED else {"shape": entry.get("shape"), "dtype": entry.get("dtype"), "severities": item["severities"]}

    # Validate labels
    labels = result["labels"]
    if not (labels.get("shape") == [50000] and labels.get("dtype") == "uint8"
            and labels.get("unique") == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            and labels.get("blocks") == 5 and labels.get("block_size") == N_PER_SEVERITY
            and labels.get("size_bytes") == EXPECTED_LABEL_FILE_SIZE):
        errors.append({"check": "labels", "labels": labels})

    # Validate archive-level fields
    archive = result["archive"]
    if archive.get("md5") != EXPECTED_MD5:
        errors.append({"check": "archive_md5", "expected": EXPECTED_MD5, "got": archive.get("md5")})
    if archive.get("size_bytes") != 2918471680:
        errors.append({"check": "archive_size", "expected": 2918471680, "got": archive.get("size_bytes")})
    if archive.get("corruption_file_count") != 19:
        errors.append({"check": "archive_corruption_file_count", "expected": 19, "got": archive.get("corruption_file_count")})

    result["errors"] = errors
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tar", nargs="?", default=None, help="path to CIFAR-10-C.tar (archive mode)")
    ap.add_argument("--manifest", default=None, help="path to verified manifest JSON (manifest mode)")
    ap.add_argument("--cifar10", default=None, help="optional canonical CIFAR-10 Python tarball (archive mode)")
    ap.add_argument("--output", default="week1_dataset_inventory.json")
    args = ap.parse_args()

    if args.manifest:
        result = inventory_from_manifest(args.manifest)
    elif args.tar:
        result = inventory_from_archive(args.tar, args.cifar10)
    else:
        ap.error("provide either a CIFAR-10-C.tar path or --manifest <manifest.json>")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({
        "mode": result.get("mode", "archive"),
        "corruption_count": result["corruption_count"],
        "corruptions": result["corruptions"],
        "classification": result["classification"],
        "required": {k: {"shape": v.get("shape"), "all_severities_10000": all(v["severities"][str(s)]["count"] == N_PER_SEVERITY for s in range(1, 6))} for k, v in result["required_corruptions"].items()},
        "labels": result["labels"],
        "label_check": result["label_check"],
        "errors": result["errors"],
    }, indent=2))
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
