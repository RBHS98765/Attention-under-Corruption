"""CIFAR-10-C evaluation loop for the four primary corruptions (Week 3, Part C).

- Four primary corruptions: brightness, contrast, defocus_blur, elastic_transform.
- Severities 1-5; the clean CIFAR-10 test set is severity 0.
- Model weights and batch-norm statistics are frozen (model.eval(), no_grad).
- No retraining or adaptation per corruption or severity.
- Reports accuracy per corruption and severity; records metrics per variant,
  seed, corruption, severity, checkpoint path, and timestamp.
"""
import json
import os
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as trn

import config
from models.backbone import build_model


def load_clean_test(device, subset=None):
    eval_tf = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD),
    ])
    ds = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=False, transform=eval_tf)
    if subset is not None and subset < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(subset)))
    return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)


def load_corruption(name, device, subset=None):
    """Load one corruption .npy [50000,32,32,3] uint8, split into 5 severity blocks.
    If subset is given, take the first `subset` images of each severity block."""
    arr = np.load(os.path.join(config.CIFAR10C_DIR, f"{name}.npy"))
    labels = np.load(os.path.join(config.CIFAR10C_DIR, "labels.npy"))
    assert arr.shape == (50000, 32, 32, 3) and arr.dtype == np.uint8
    assert labels.shape == (50000,)
    n = subset or 10000
    mean = torch.tensor(config.NORMALIZE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(config.NORMALIZE_STD).view(1, 3, 1, 1)
    tensors = []
    for sev in config.SEVERITIES:
        sl = (sev - 1) * 10000
        x = torch.from_numpy(arr[sl:sl + n].transpose(0, 3, 1, 2)).float() / 255.0
        x = (x - mean) / std
        y = torch.from_numpy(labels[sl:sl + n]).long()
        tensors.append((x, y))
    return tensors


@torch.no_grad()
def accuracy(model, x, y, device):
    model.eval()
    x, y = x.to(device), y.to(device)
    out = model(x)
    return (out.argmax(dim=1) == y).float().mean().item(), x.size(0)


def evaluate_variant(variant, seed=None, ckpt_path=None, device=None):
    """Evaluate one variant on clean CIFAR-10 (sev 0) and the 4 primary corruptions."""
    device = device or torch.device("cpu")
    seed = config.SEED if seed is None else seed
    model = build_model(attention_type=variant).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd["model_state_dict"])
        loaded_epoch = sd.get("epoch")
    else:
        loaded_epoch = None

    results = {"variant": variant, "seed": seed, "ckpt_path": ckpt_path,
               "loaded_epoch": loaded_epoch, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "corruptions": {}}

    # severity 0: clean CIFAR-10 test set
    clean_loader = load_clean_test(device, subset=config.EVAL_CLEAN_SUBSET)
    correct, total = 0, 0
    for x, y in clean_loader:
        acc, n = accuracy(model, x, y, device)
        correct += acc * n
        total += n
    results["corruptions"]["clean"] = {
        "severity": 0, "accuracy": correct / max(total, 1), "num_examples": total}

    # severities 1-5 for each primary corruption
    for name in config.PRIMARY_CORRUPTIONS:
        blocks = load_corruption(name, device, subset=config.EVAL_CORRUPTION_SUBSET)
        for sev, (x, y) in zip(config.SEVERITIES, blocks):
            acc, n = accuracy(model, x, y, device)
            results["corruptions"].setdefault(name, {})[sev] = {
                "accuracy": acc, "num_examples": n}
    return results


def summarize(results):
    """Return a compact summary dict for the report."""
    out = {"variant": results["variant"], "clean_acc": results["corruptions"]["clean"]["accuracy"]}
    for name in config.PRIMARY_CORRUPTIONS:
        accs = [results["corruptions"][name][s]["accuracy"] for s in config.SEVERITIES]
        out[name] = {"sev1_5_acc": accs, "mean": float(np.mean(accs))}
    return out


if __name__ == "__main__":
    import sys
    variant = sys.argv[1] if len(sys.argv) > 1 else "none"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    res = evaluate_variant(variant, ckpt_path=ckpt)
    print(json.dumps(summarize(res), indent=2))
