import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import QualityRegressionDataset
from models import get_quality_model


DEFAULT_TRAIN_CSV = "data/splits/ranking_train.csv"
DEFAULT_VAL_CSV = "data/splits/ranking_val.csv"
DEFAULT_REFERENCE_CSV = "data/annotations/ranking_reference_top100.csv"
DEFAULT_IMAGE_DIR = "data/raw"
DEFAULT_CHECKPOINT_DIR = "outputs/checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an image-quality regression model for Unit 7."
    )
    parser.add_argument("--config", required=True, help="Path to a ranking YAML config.")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", default=DEFAULT_VAL_CSV)
    parser.add_argument(
        "--reference-csv",
        default=DEFAULT_REFERENCE_CSV,
        help="Locked Top-100 file used only for leakage validation, never training.",
    )
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--project", default="training-Unit7-Ranking")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Use 'disabled' for a local test that should not create a W&B run.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use at most 16 train/validation images for two epochs.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    required = {
        "model_name",
        "pretrained",
        "learning_rate",
        "batch_size",
        "epochs",
        "optimizer",
        "loss",
        "input_size",
        "weight_decay",
        "random_seed",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Config is missing required keys: {', '.join(sorted(missing))}")
    if config["model_name"] not in {"vgg16", "resnet18", "resnet50"}:
        raise ValueError(f"Unsupported model_name: {config['model_name']}")
    if config["optimizer"].lower() != "adamw":
        raise ValueError("This training script currently supports optimizer: AdamW")
    if config["loss"].lower() != "smoothl1loss":
        raise ValueError("This training script currently supports loss: SmoothL1Loss")
    return config


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings make repeated runs easier to compare.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(input_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, val_transform


def validate_split_integrity(train_csv, val_csv, reference_csv):
    split_paths = {
        "train": train_csv,
        "validation": val_csv,
        "locked reference": reference_csv,
    }
    image_id_sets = {}

    for split_name, path in split_paths.items():
        frame = pd.read_csv(path)
        if "image_id" not in frame.columns:
            raise ValueError(f"{path} must contain an image_id column")
        if frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
            raise ValueError(f"{path} contains missing or duplicate image_id values")
        image_id_sets[split_name] = set(frame["image_id"].astype(str))

    pairs = (
        ("train", "validation"),
        ("train", "locked reference"),
        ("validation", "locked reference"),
    )
    for left_name, right_name in pairs:
        overlap = image_id_sets[left_name].intersection(image_id_sets[right_name])
        if overlap:
            examples = ", ".join(sorted(overlap)[:10])
            raise ValueError(
                f"Data leakage: {left_name} and {right_name} overlap "
                f"on {len(overlap)} image(s), including {examples}"
            )


def build_loaders(config, args, device):
    validate_split_integrity(args.train_csv, args.val_csv, args.reference_csv)
    train_transform, val_transform = build_transforms(config["input_size"])
    train_dataset = QualityRegressionDataset(
        args.train_csv, args.image_dir, transform=train_transform
    )
    val_dataset = QualityRegressionDataset(
        args.val_csv, args.image_dir, transform=val_transform
    )

    if args.smoke_test:
        train_size = min(16, len(train_dataset))
        val_size = min(16, len(val_dataset))
        train_dataset = Subset(train_dataset, range(train_size))
        val_dataset = Subset(val_dataset, range(val_size))
        config["epochs"] = 2

    num_workers = int(config.get("num_workers", 4))
    loader_options = {
        "batch_size": int(config["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    return train_loader, val_loader


def regression_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    correlation = spearmanr(targets, predictions).statistic
    spearman = 0.0 if correlation is None or not np.isfinite(correlation) else float(correlation)
    return {"mae": mae, "rmse": rmse, "spearman": spearman}


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_examples = 0
    all_targets = []
    all_predictions = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(images).squeeze(1)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = regression_metrics(all_targets, all_predictions)
    metrics["loss"] = total_loss / total_examples
    return metrics


def is_better_checkpoint(val_metrics, best_spearman, best_mae):
    tolerance = 1e-12
    if val_metrics["spearman"] > best_spearman + tolerance:
        return True
    return (
        math.isclose(val_metrics["spearman"], best_spearman, abs_tol=tolerance)
        and val_metrics["mae"] < best_mae
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    set_random_seed(int(config["random_seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(config, args, device)

    model = get_quality_model(
        model_name=config["model_name"],
        pretrained=bool(config["pretrained"]),
    ).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_prefix = "ranking_smoke_best" if args.smoke_test else "ranking_best"
    checkpoint_path = os.path.join(
        args.checkpoint_dir, f"{checkpoint_prefix}_{config['model_name']}.pth"
    )

    run_name = (
        f"{config['model_name']}_quality_lr{config['learning_rate']}"
        f"_bs{config['batch_size']}"
    )
    if args.smoke_test:
        run_name = f"SMOKE_TEST_{run_name}"

    wandb_config = dict(config)
    wandb_config.update(
        {
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "locked_reference_csv": args.reference_csv,
            "checkpoint_selection": "highest val/spearman; tie-break lowest val/mae",
            "smoke_test": args.smoke_test,
            "device": str(device),
        }
    )
    run = wandb.init(
        project=args.project,
        name=run_name,
        config=wandb_config,
        mode=args.wandb_mode,
    )
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("learning_rate", step_metric="epoch")
    run.define_metric("val/spearman", summary="max")
    run.define_metric("val/mae", summary="min")
    run.define_metric("val/rmse", summary="min")
    run.define_metric("val/loss", summary="min")

    print(f"Device: {device}")
    print(f"Model: {config['model_name']}")
    print(f"Training examples: {len(train_loader.dataset)}")
    print(f"Validation examples: {len(val_loader.dataset)}")
    print(f"Checkpoint: {checkpoint_path}")

    best_spearman = -float("inf")
    best_mae = float("inf")
    best_epoch = 0

    try:
        for epoch in range(1, int(config["epochs"]) + 1):
            train_metrics = run_epoch(
                model, train_loader, criterion, device, optimizer=optimizer
            )
            val_metrics = run_epoch(model, val_loader, criterion, device)

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/mae": train_metrics["mae"],
                    "train/rmse": train_metrics["rmse"],
                    "train/spearman": train_metrics["spearman"],
                    "val/loss": val_metrics["loss"],
                    "val/mae": val_metrics["mae"],
                    "val/rmse": val_metrics["rmse"],
                    "val/spearman": val_metrics["spearman"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            print(
                f"Epoch [{epoch}/{config['epochs']}] "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"MAE={train_metrics['mae']:.4f}, "
                f"Spearman={train_metrics['spearman']:.4f} | "
                f"Val Loss={val_metrics['loss']:.4f}, "
                f"MAE={val_metrics['mae']:.4f}, "
                f"RMSE={val_metrics['rmse']:.4f}, "
                f"Spearman={val_metrics['spearman']:.4f}"
            )

            if is_better_checkpoint(val_metrics, best_spearman, best_mae):
                best_spearman = val_metrics["spearman"]
                best_mae = val_metrics["mae"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": config["model_name"],
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "config": config,
                    },
                    checkpoint_path,
                )
                print(
                    f"Saved best checkpoint at epoch {epoch}: "
                    f"Spearman={best_spearman:.4f}, MAE={best_mae:.4f}"
                )

        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_spearman"] = best_spearman
        run.summary["best_val_mae"] = best_mae
        run.summary["checkpoint_path"] = checkpoint_path
    finally:
        wandb.finish()

    print(
        f"Training complete. Best epoch={best_epoch}, "
        f"Val Spearman={best_spearman:.4f}, Val MAE={best_mae:.4f}"
    )


if __name__ == "__main__":
    main()
