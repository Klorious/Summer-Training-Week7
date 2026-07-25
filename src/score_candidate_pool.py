import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from src.models import get_quality_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POOL_CSV = "data/splits/candidate_pool.csv"
OUTPUT_DIR = "outputs/ranking"
CHECKPOINT_DIR = "outputs/checkpoints"
MODELS = ["vgg16", "resnet18", "resnet50"]

EXPECTED_POOL_SIZE = 500
RANDOM_SEED = 20260725
TTA_RUNS = 5
PENALTY_WEIGHT = 0.5

MODEL_SCORE_COLUMNS = [
    "vgg16_raw_score",
    "vgg16_adjusted_score",
    "resnet18_raw_score",
    "resnet18_adjusted_score",
    "resnet50_raw_score",
    "resnet50_adjusted_score",
]
ENSEMBLE_SCORE_COLUMNS = [
    "ensemble_raw_score",
    "ensemble_adjusted_score",
]
ALL_SCORE_COLUMNS = MODEL_SCORE_COLUMNS + ENSEMBLE_SCORE_COLUMNS


# Raw inference must use the same preprocessing as ranking validation.
base_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Confidence adjustment uses input-level test-time augmentation (TTA).
# The model itself remains in eval mode so BatchNorm statistics are never changed.
tta_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CandidateDataset(torch.utils.data.Dataset):
    REQUIRED_COLUMNS = {"image_id", "image_path"}

    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        missing_columns = self.REQUIRED_COLUMNS.difference(self.data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_file} is missing required columns: {missing}")

        if len(self.data) != EXPECTED_POOL_SIZE:
            raise ValueError(
                f"Expected {EXPECTED_POOL_SIZE} candidate images, "
                f"but {csv_file} contains {len(self.data)}"
            )
        if self.data["image_id"].isna().any():
            raise ValueError(f"{csv_file} contains missing image_id values")
        if self.data["image_id"].duplicated().any():
            duplicates = self.data.loc[
                self.data["image_id"].duplicated(), "image_id"
            ].astype(str)
            raise ValueError(
                f"{csv_file} contains duplicate image_id values: "
                f"{duplicates.head(10).tolist()}"
            )
        if self.data["image_path"].isna().any():
            raise ValueError(f"{csv_file} contains missing image_path values")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = str(row["image_path"])
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        return (
            str(row["image_id"]),
            image_path,
            self.transform(image),
            image,
        )


def get_z_scores(scores, score_name):
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{score_name} contains NaN or infinite values")

    standard_deviation = float(np.std(values))
    if standard_deviation <= 1e-12:
        raise ValueError(
            f"{score_name} is constant and cannot be standardized for ensemble scoring"
        )
    return (values - np.mean(values)) / standard_deviation


def validate_score_frame(frame):
    required_columns = {"image_id", "image_path", *ALL_SCORE_COLUMNS}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Final score table is missing columns: {missing}")

    if len(frame) != EXPECTED_POOL_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_POOL_SIZE} score rows, but found {len(frame)}"
        )
    if frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
        raise ValueError("Final score table has missing or duplicate image_id values")

    numeric_scores = frame[ALL_SCORE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric_scores.isna().any().any():
        bad_columns = numeric_scores.columns[numeric_scores.isna().any()].tolist()
        raise ValueError(f"NaN or non-numeric scores found in: {bad_columns}")
    if not np.isfinite(numeric_scores.to_numpy(dtype=np.float64)).all():
        raise ValueError("Infinite values found in final score table")


def load_model(model_name):
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"ranking_best_{model_name}.pth"
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_quality_model(model_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, checkpoint_path


def main():
    set_random_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"TTA runs: {TTA_RUNS}")
    print(f"Confidence penalty weight: {PENALTY_WEIGHT}")

    dataset = CandidateDataset(POOL_CSV, transform=base_transform)
    pool_frame = dataset.data[["image_id", "image_path"]].copy()
    results_frame = pool_frame.copy()
    print(f"Candidate pool size: {len(dataset)}")

    for model_name in MODELS:
        print(f"\n[{model_name}] Loading checkpoint and scoring candidate pool")
        model, checkpoint_path = load_model(model_name)
        print(f"Checkpoint: {checkpoint_path}")

        raw_scores = []
        adjusted_scores = []

        for idx in tqdm(range(len(dataset)), desc=f"{model_name} inference"):
            _, _, base_tensor, source_image = dataset[idx]

            # Raw quality score: deterministic validation preprocessing.
            with torch.inference_mode():
                raw_score = model(
                    base_tensor.unsqueeze(0).to(DEVICE, non_blocking=True)
                ).item()
            raw_scores.append(float(raw_score))

            # Reset the seed per image so every model receives the same five
            # deterministic TTA views for that image.
            image_seed = RANDOM_SEED + idx
            set_random_seed(image_seed)

            tta_predictions = []
            for _ in range(TTA_RUNS):
                tta_tensor = tta_transform(source_image).unsqueeze(0).to(
                    DEVICE, non_blocking=True
                )
                with torch.inference_mode():
                    tta_predictions.append(float(model(tta_tensor).item()))

            tta_mean = float(np.mean(tta_predictions))
            tta_std = float(np.std(tta_predictions))
            adjusted_score = tta_mean - PENALTY_WEIGHT * tta_std
            adjusted_scores.append(float(adjusted_score))

        raw_column = f"{model_name}_raw_score"
        adjusted_column = f"{model_name}_adjusted_score"
        results_frame[raw_column] = raw_scores
        results_frame[adjusted_column] = adjusted_scores

        model_frame = pool_frame.copy()
        model_frame["raw_score"] = raw_scores
        model_frame["adjusted_score"] = adjusted_scores

        model_scores = model_frame[["raw_score", "adjusted_score"]].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(model_scores).all():
            raise ValueError(f"{model_name} produced NaN or infinite scores")

        model_output_path = os.path.join(
            OUTPUT_DIR, f"{model_name}_scores.csv"
        )
        model_frame.to_csv(model_output_path, index=False)
        print(
            f"{model_name} raw range: "
            f"{min(raw_scores):.4f} to {max(raw_scores):.4f}"
        )
        print(
            f"{model_name} adjusted range: "
            f"{min(adjusted_scores):.4f} to {max(adjusted_scores):.4f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Lock the original ensemble definition: average of per-model z-scores.
    raw_z_scores = [
        get_z_scores(
            results_frame[f"{model_name}_raw_score"],
            f"{model_name}_raw_score",
        )
        for model_name in MODELS
    ]
    adjusted_z_scores = [
        get_z_scores(
            results_frame[f"{model_name}_adjusted_score"],
            f"{model_name}_adjusted_score",
        )
        for model_name in MODELS
    ]
    results_frame["ensemble_raw_score"] = np.mean(raw_z_scores, axis=0)
    results_frame["ensemble_adjusted_score"] = np.mean(
        adjusted_z_scores, axis=0
    )

    validate_score_frame(results_frame)

    print("\nFinal score ranges:")
    for column in ALL_SCORE_COLUMNS:
        print(
            f"{column}: min={results_frame[column].min():.4f}, "
            f"max={results_frame[column].max():.4f}"
        )

    final_output_path = os.path.join(OUTPUT_DIR, "all_model_scores.csv")
    results_frame.to_csv(final_output_path, index=False)
    print(f"\nSaved complete score table: {final_output_path}")
    print(
        f"Validation passed: {len(results_frame)} unique images, "
        "all scores numeric and finite"
    )


if __name__ == "__main__":
    main()
