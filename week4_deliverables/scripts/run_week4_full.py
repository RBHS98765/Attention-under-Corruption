"""Week 4 full one-seed experiment runner.

This runner is intentionally separate from the Week 3 smoke runner. It uses:
  - config.EPOCHS = 100;
  - the full CIFAR-10 training split (TRAIN_SUBSET=None);
  - batch size 128, SGD, cosine annealing over all steps;
  - seed 0;
  - full clean test evaluation and all 10,000 examples per severity for each
    primary corruption.

Run on a machine with adequate compute:
    python3 scripts/run_week4_full.py

It does NOT run the three-seed experiment. It writes final checkpoints,
training CSVs, clean-evaluation JSON/CSV, corruption-evaluation JSON/CSV, and
result JSON files after each completed variant.
"""
import csv
import json
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import config
import train
import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
LOGS = config.LOG_DIR
CKPTS = config.CKPT_DIR
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
os.makedirs(CKPTS, exist_ok=True)


def write_eval_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    torch.set_num_threads(config.NUM_THREADS)
    clean_rows = []
    corruption_rows = []
    clean_results = {"seed": config.SEED, "variants": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    corr_results = {"seed": config.SEED, "variants": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    for variant in config.ATTENTION_TYPES:
        print(f"===== WEEK 4 TRAIN {variant} =====", flush=True)
        model, metrics = train.train_variant(
            variant=variant, epochs=config.EPOCHS, subset=None,
            seed=config.SEED, batch_size=config.BATCH_SIZE,
            ckpt_dir=CKPTS, log_dir=LOGS, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")	)
        # Rename/copy Week 3-compatible training log to Week 4 contract.
        src_log = metrics["log_path"]
        dst_log = os.path.join(LOGS, f"train_{variant}_seed{config.SEED}.csv")
        shutil.copyfile(src_log, dst_log)

        result = evaluate.evaluate_variant(
            variant=variant, seed=config.SEED, ckpt_path=metrics["ckpt_path"],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        clean = result["corruptions"]["clean"]
        clean_results["variants"][variant] = {
            "accuracy": clean["accuracy"], "num_examples": clean["num_examples"],
            "checkpoint_path": metrics["ckpt_path"], "seed": config.SEED,
            "timestamp": result["timestamp"],
        }
        clean_rows.append({"variant": variant, **clean_results["variants"][variant]})
        corr_results["variants"][variant] = result
        for corr in config.PRIMARY_CORRUPTIONS:
            for sev in config.SEVERITIES:
                row = result["corruptions"][corr][sev]
                corruption_rows.append({
                    "variant": variant, "corruption": corr, "severity": sev,
                    "accuracy": row["accuracy"], "num_examples": row["num_examples"],
                    "checkpoint_path": metrics["ckpt_path"], "seed": config.SEED,
                    "timestamp": result["timestamp"],
                })
        json.dump(clean_results, open(os.path.join(RESULTS, "week4_clean_results.json"), "w"), indent=2)
        json.dump(corr_results, open(os.path.join(RESULTS, "week4_corruption_results.json"), "w"), indent=2)
        write_eval_rows(os.path.join(LOGS, f"eval_{variant}_seed{config.SEED}_clean.csv"), [clean_rows[-1]])
        write_eval_rows(os.path.join(LOGS, f"eval_{variant}_seed{config.SEED}_corruption.csv"), [r for r in corruption_rows if r["variant"] == variant])

    write_eval_rows(os.path.join(LOGS, "week4_clean_all_variants.csv"), clean_rows)
    write_eval_rows(os.path.join(LOGS, "week4_corruption_all_variants.csv"), corruption_rows)
    print("WEEK 4 FULL ONE-SEED RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
