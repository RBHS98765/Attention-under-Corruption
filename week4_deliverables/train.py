
"""CIFAR-10 training loop for the matched backbone variants (Week 3, Part B).

- Standard CIFAR-10 training split with a standard augmentation pipeline.
- Cross-entropy loss, SGD with momentum + weight decay, cosine-annealing LR.
- Deterministic seeding.
- Checkpoint saved after every epoch.
- CSV logging after every epoch.
- No test-time adaptation and no corruption-specific training.
"""

import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as trn

import config
from models.backbone import build_model


def set_seed(seed):
    """Deterministic seeding."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms():
    train_tf = trn.Compose([
        trn.RandomCrop(32, padding=4),
        trn.RandomHorizontalFlip(),
        trn.ToTensor(),
        trn.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD),
    ])

    eval_tf = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD),
    ])

    return train_tf, eval_tf


def load_cifar10(subset=None):
    """Load CIFAR-10."""
    train_tf, eval_tf = get_transforms()

    train_ds = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=False,
        transform=train_tf
    )

    test_ds = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=False,
        transform=eval_tf
    )

    if subset is not None and subset < len(train_ds):
        idx = list(range(subset))
        train_ds = torch.utils.data.Subset(train_ds, idx)

    return train_ds, test_ds


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    variant,
    epoch,
    log_rows
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)

    lr = optimizer.param_groups[0]["lr"]

    log_rows.append({
        "variant": variant,
        "epoch": epoch,
        "train_loss": avg_loss,
        "train_acc": acc,
        "lr": lr,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    return avg_loss, acc, lr


def evaluate(model, loader, device, criterion=None):
    """Evaluate on the supplied CIFAR-10 evaluation loader."""
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)

            if criterion is not None:
                loss_sum += criterion(out, y).item() * x.size(0)

            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return (
        correct / max(total, 1),
        loss_sum / max(total, 1)
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    seed,
    variant,
    metrics
):
    torch.save({
        "variant": variant,
        "seed": seed,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            k: v
            for k, v in vars(config).items()
            if k.isupper() and not k.startswith("__")
        },
        "metrics": metrics,
    }, path)


def write_log_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        w.writeheader()
        w.writerows(rows)


def train_variant(
    variant,
    epochs=None,
    subset=None,
    seed=None,
    batch_size=None,
    lr=None,
    ckpt_dir=None,
    log_dir=None,
    device=None
):
    """Train one attention variant and return its final metrics."""

    epochs = config.EPOCHS if epochs is None else epochs
    subset = config.TRAIN_SUBSET if subset is None else subset
    seed = config.SEED if seed is None else seed
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    lr = config.LEARNING_RATE if lr is None else lr

    # Choose a separate checkpoint directory for each variant.
    if ckpt_dir is None:
        ckpt_dirs = {
            "none": "C:/cifar_data/week4_ckpts_none",
            "se": "C:/cifar_data/week4_ckpts_se",
            "bam": "C:/cifar_data/week4_ckpts_bam",
            "cbam": "C:/cifar_data/week4_ckpts_cbam",
        }
        ckpt_dir = ckpt_dirs.get(variant, config.CKPT_DIR)

    log_dir = config.LOG_DIR if log_dir is None else log_dir
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(seed)

    train_ds, test_ds = load_cifar10(subset=subset)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = build_model(
        attention_type=variant
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    # Cosine annealing over the full epoch schedule.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    log_rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, cur_lr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            variant,
            epoch,
            log_rows
        )

        # This currently evaluates the CIFAR-10 test split each epoch.
        # For final research reporting, keep a separate validation split
        # for model selection and reserve the test split for final evaluation.
        val_acc, val_loss = evaluate(
            model,
            test_loader,
            device,
            criterion
        )

        log_rows[-1]["val_loss"] = val_loss
        log_rows[-1]["val_acc"] = val_acc

        # Save a checkpoint after every epoch.
        ckpt_path = os.path.join(
            ckpt_dir,
            config.CKPT_NAME.format(
                variant=variant,
                seed=seed,
                epoch=epoch
            )
        )

        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            epoch,
            seed,
            variant,
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )

        print(
            f"[{variant}] epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"lr={cur_lr:.5f}"
        )

        # Step once per epoch.
        scheduler.step()

    # Save complete training history.
    log_path = os.path.join(
        log_dir,
        f"train_{variant}_seed{seed}.csv"
    )

    write_log_csv(log_path, log_rows)

    final_ckpt_path = os.path.join(
        ckpt_dir,
        config.CKPT_NAME.format(
            variant=variant,
            seed=seed,
            epoch=epochs
        )
    )

    return model, {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "ckpt_path": final_ckpt_path,
        "log_path": log_path,
        "seed": seed,
        "epochs": epochs,
        "variant": variant,
    }


if __name__ == "__main__":
    import sys

    variant = sys.argv[1] if len(sys.argv) > 1 else "none"

    m, metrics = train_variant(variant)

    print(json.dumps(metrics, indent=2))
