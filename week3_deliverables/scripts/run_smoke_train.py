"""Week 3 short training smoke test (Part D).

Usage:
  python3 scripts/run_smoke_train.py            # run all variants
  python3 scripts/run_smoke_train.py --variant none|se|bam|cbam   # one variant

For every variant:
  - train at most one epoch on a 10,000-image CIFAR-10 subset, batch 128, seed 0;
  - confirm finite loss and reasonable accuracy (not exactly 0 or 1);
  - save a checkpoint;
  - evaluate on clean CIFAR-10 (severity 0, subset) and the four primary
    corruptions (brightness, contrast, defocus_blur, elastic_transform) at
    severities 1-5 (subset per severity);
  - write per-variant CSV logs and accumulate a consolidated JSON result.

Exits non-zero if any check fails. Smoke-test accuracy is NOT an experimental
result; it only validates the pipeline.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import config
import train as train_mod
import evaluate as eval_mod

RESULTS_PATH = os.path.join(config.LOG_DIR, "week3_smoke_results.json")


def run_variant(variant, seed):
    print(f"\n========== SMOKE TRAIN: {variant} ==========", flush=True)
    entry = {"seed": seed, "variant": variant}
    try:
        model, metrics = train_mod.train_variant(
            variant, epochs=1, subset=config.TRAIN_SUBSET, seed=seed,
            batch_size=config.BATCH_SIZE, device=torch.device("cpu"))
    except Exception as e:
        print(f"  FAIL [{variant}] training error: {e!r}", flush=True)
        entry["error"] = repr(e)
        entry["ok"] = False
        return entry

    loss_ok = bool(torch.isfinite(torch.tensor(metrics["train_loss"])))
    acc = metrics["train_acc"]
    acc_ok = 0.0 < acc < 1.0
    ckpt_ok = os.path.exists(metrics["ckpt_path"]) and os.path.getsize(metrics["ckpt_path"]) > 0

    try:
        res = eval_mod.evaluate_variant(variant, seed=seed,
                                        ckpt_path=metrics["ckpt_path"],
                                        device=torch.device("cpu"))
        clean_acc = res["corruptions"]["clean"]["accuracy"]
        eval_ok = 0.0 < clean_acc < 1.0
        corruption_ok = all(name in res["corruptions"] for name in config.PRIMARY_CORRUPTIONS)
    except Exception as e:
        print(f"  FAIL [{variant}] evaluation error: {e!r}", flush=True)
        entry["error"] = repr(e)
        entry["ok"] = False
        return entry

    variant_ok = loss_ok and acc_ok and ckpt_ok and eval_ok and corruption_ok
    entry.update({
        "train_loss": metrics["train_loss"],
        "train_acc": acc,
        "val_acc": metrics["val_acc"],
        "loss_finite": loss_ok,
        "acc_reasonable": acc_ok,
        "ckpt_saved": ckpt_ok,
        "ckpt_path": metrics["ckpt_path"],
        "clean_acc": clean_acc,
        "eval_ok": eval_ok,
        "corruption_eval_ok": corruption_ok,
        "evaluation": eval_mod.summarize(res),
        "ok": variant_ok,
    })
    print(f"  [{variant}] train_loss={metrics['train_loss']:.4f} "
          f"train_acc={acc:.4f} clean_acc={clean_acc:.4f} "
          f"loss_finite={loss_ok} acc_ok={acc_ok} ckpt={ckpt_ok} eval_ok={eval_ok}", flush=True)
    return entry


def main():
    torch.set_num_threads(config.NUM_THREADS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=config.ATTENTION_TYPES, default=None)
    args = parser.parse_args()

    seed = config.SEED
    # load existing accumulated results if any
    summary = {"seed": seed, "variants": {}, "ok": True}
    if os.path.exists(RESULTS_PATH):
        try:
            summary = json.load(open(RESULTS_PATH))
        except Exception:
            summary = {"seed": seed, "variants": {}, "ok": True}

    targets = [args.variant] if args.variant else config.ATTENTION_TYPES
    for variant in targets:
        summary["variants"][variant] = run_variant(variant, seed)
        if not summary["variants"][variant].get("ok", False):
            summary["ok"] = False
        # save after each variant so a reset loses at most the current one
        os.makedirs(config.LOG_DIR, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(summary, f, indent=2)

    done = set(summary["variants"].keys())
    all_done = set(config.ATTENTION_TYPES).issubset(done)
    all_ok = summary["ok"]
    print(f"\nVariants done: {sorted(done)}")
    print(f"Wrote: {RESULTS_PATH}")
    print("STATUS:", "PASS" if (all_done and all_ok) else ("PARTIAL" if all_done else "INCOMPLETE"))
    sys.exit(0 if (all_done and all_ok) else 1)


if __name__ == "__main__":
    main()
