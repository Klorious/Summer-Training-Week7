import argparse
import csv
import math
from pathlib import Path

import wandb


FORMAL_RUNS = {
    "VGG16": "rwu768x7",
    "ResNet18": "g409y4vj",
    "ResNet50": "ijwx6i2k",
}

HISTORY_KEYS = [
    "epoch",
    "val/mae",
    "val/rmse",
    "val/spearman",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export the exact validation MAE, RMSE, and Spearman values at "
            "the checkpoint-selected best epoch for the three formal "
            "advanced ranking runs."
        )
    )
    parser.add_argument("--entity", default="klorius-")
    parser.add_argument("--project", default="training-Unit7-Ranking")
    parser.add_argument(
        "--output",
        default="outputs/report_data/ranking_best_validation_metrics.csv",
    )
    return parser.parse_args()


def is_finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def select_best_history_row(run):
    best_epoch = run.summary.get("best_epoch")
    if best_epoch is None:
        raise RuntimeError(f"{run.id}: W&B summary has no best_epoch")
    best_epoch = int(best_epoch)

    history_rows = list(
        run.scan_history(keys=HISTORY_KEYS, page_size=1000)
    )
    matching_rows = []
    for row in history_rows:
        epoch = row.get("epoch")
        if not is_finite_number(epoch) or int(float(epoch)) != best_epoch:
            continue
        if not all(is_finite_number(row.get(key)) for key in HISTORY_KEYS[1:]):
            continue
        matching_rows.append(row)

    if not matching_rows:
        raise RuntimeError(
            f"{run.id}: no complete validation row found at epoch {best_epoch}"
        )

    # Metrics were logged once per epoch. Taking the last matching row also
    # handles an accidentally duplicated epoch log deterministically.
    return best_epoch, matching_rows[-1]


def main():
    args = parse_args()
    api = wandb.Api()
    records = []

    for model_name, run_id in FORMAL_RUNS.items():
        run_path = f"{args.entity}/{args.project}/{run_id}"
        run = api.run(run_path)
        best_epoch, row = select_best_history_row(run)

        record = {
            "model": model_name,
            "run_id": run_id,
            "run_name": run.name,
            "best_epoch": best_epoch,
            "val_mae": float(row["val/mae"]),
            "val_rmse": float(row["val/rmse"]),
            "val_spearman": float(row["val/spearman"]),
            "run_url": run.url,
        }
        records.append(record)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print("\nExact values at each checkpoint-selected best epoch:\n")
    for record in records:
        print(
            f"{record['model']}: "
            f"best_epoch={record['best_epoch']}, "
            f"val_mae={record['val_mae']:.6f}, "
            f"val_rmse={record['val_rmse']:.6f}, "
            f"val_spearman={record['val_spearman']:.6f}"
        )

    print("\nLaTeX rows for report/main.tex:\n")
    for record in records:
        print(
            f"{record['model']} & {record['best_epoch']} & "
            f"{record['val_mae']:.4f} & {record['val_rmse']:.4f} & "
            f"{record['val_spearman']:.4f} & "
            f"\\texttt{{{record['run_id']}}} \\\\"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
