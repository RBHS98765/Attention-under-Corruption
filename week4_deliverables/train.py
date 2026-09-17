"""CIFAR-10 training loop for the matched backbone variants (Week 3, Part B).

- Standard CIFAR-10 training split with a standard augmentation pipeline.
- Cross-entropy loss, SGD with momentum + weight decay, cosine-annealing LR.
- Deterministic seeding (set_seed), checkpointing (weights + optimizer +
  epoch + seed + config + metrics), and CSV/JSON logging.
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
        trn.Normalize(
            config.NORMALIZE_MEAN,
            config.NORMALIZE_STD
        ),
    ])

    eval_tf = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(
            config.NORMALIZE_MEAN,
            config.NORMALIZE_STD
        ),
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
        train_ds = torch.utils.data.Subset(
            train_ds,
            idx
        )

    return train_ds, test_ds


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    variant,
    epoch,
    log_rows,
    lr_scheduler=None
):
    """Train the model for one epoch."""

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * x.size(0)

        pred = out.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += x.size(0)

        if lr_scheduler is not None:
            lr_scheduler.step()

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
    """Evaluate the model on clean CIFAR-10 test data."""

    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x)

            if criterion is not None:
                loss_sum += (
                    criterion(out, y).item() * x.size(0)
                )

            correct += (
                out.argmax(dim=1) == y
            ).sum().item()

            total += x.size(0)

    accuracy = correct / max(total, 1)
    average_loss = loss_sum / max(total, 1)

    return accuracy, average_loss


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    seed,
    variant,
    metrics
):
    """Save one checkpoint."""

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
    """Write training history to CSV."""

    if not rows:
        return

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


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
    """Train one variant and save a checkpoint after every epoch."""

    epochs = epochs or config.EPOCHS
    subset = subset or config.TRAIN_SUBSET
    seed = config.SEED if seed is None else seed
    batch_size = batch_size or config.BATCH_SIZE
    lr = lr or config.LEARNING_RATE
    ckpt_dir = ckpt_dir or config.CKPT_DIR
    log_dir = log_dir or config.LOG_DIR

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = device or torch.device("cpu")

    set_seed(seed)

    train_ds, test_ds = load_cifar10(
        subset=subset
    )

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

    steps_per_epoch = len(train_loader)

    total_steps = steps_per_epoch * epochs

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )

    log_rows = []

    for epoch in range(1, epochs + 1):

        train_loss, train_acc, cur_lr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            variant=variant,
            epoch=epoch,
            log_rows=log_rows,
            lr_scheduler=scheduler
        )

        val_acc, val_loss = evaluate(
            model,
            test_loader,
            device,
            criterion
        )

        log_rows[-1]["val_loss"] = val_loss
        log_rows[-1]["val_acc"] = val_acc

        print(
            f"[{variant}] epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"lr={cur_lr:.5f}"
        )

        # Save checkpoint after EVERY epoch.
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
                "val_loss": val_loss
            }
        )

    # Save a unique CSV for every variant + seed.
    log_path = os.path.join(
        log_dir,
        f"train_{variant}_seed{seed}.csv"
    )

    write_log_csv(
        log_path,
        log_rows
    )

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
        "variant": variant
    }


if __name__ == "__main__":

    import sys

    variant = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "none"
    )

    m, metrics = train_variant(variant)

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )
