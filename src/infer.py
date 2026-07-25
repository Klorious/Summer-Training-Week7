import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import get_binary_model


REFERENCE_CSV = "data/annotations/binary_reference_100.csv"
IMG_DIR = "data/raw"
OUTPUT_CSV = "outputs/predictions/reference_results.csv"

# The ResNet18 checkpoint is intentionally learning-rate-specific because
# validation F1 selected LR=0.0001 over LR=0.001.
MODEL_SPECS = {
    "VGG16": {
        "model_name": "vgg16",
        "checkpoint": "outputs/checkpoints/best_vgg16.pth",
        "learning_rate": 0.0001,
        "source_wandb_run_id": "czw8hak1",
    },
    "ResNet18": {
        "model_name": "resnet18",
        "checkpoint": "outputs/checkpoints/best_resnet18_lr0.0001.pth",
        "learning_rate": 0.0001,
        "source_wandb_run_id": None,
    },
    "ResNet50": {
        "model_name": "resnet50",
        "checkpoint": "outputs/checkpoints/best_resnet50.pth",
        "learning_rate": 0.0001,
        "source_wandb_run_id": "0zwvcvtq",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the three selected binary models on the locked reference set."
    )
    parser.add_argument("--reference-csv", default=REFERENCE_CSV)
    parser.add_argument("--image-dir", default=IMG_DIR)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    """Load both legacy raw state_dict files and new metadata checkpoints."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with PyTorch releases that do not expose weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = {
            "learning_rate": checkpoint.get("learning_rate"),
            "source_wandb_run_id": checkpoint.get("wandb_run_id"),
            "source_wandb_run_url": checkpoint.get("wandb_run_url"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_val_f1": checkpoint.get("best_val_f1"),
        }
    else:
        model.load_state_dict(checkpoint)
        metadata = {}
    return metadata


def validate_reference_table(data_frame, csv_path):
    required = {"image_id", "image_path", "human_label"}
    missing = required.difference(data_frame.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns: {', '.join(sorted(missing))}"
        )
    if len(data_frame) != 100:
        raise ValueError(f"Expected 100 reference images, found {len(data_frame)}")
    if data_frame["image_id"].duplicated().any():
        duplicates = data_frame.loc[
            data_frame["image_id"].duplicated(), "image_id"
        ].tolist()
        raise ValueError(f"Duplicate reference image IDs: {duplicates[:10]}")
    invalid_labels = sorted(set(data_frame["human_label"]) - {"Good", "Bad"})
    if invalid_labels:
        raise ValueError(f"Invalid human labels: {invalid_labels}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    reference_df = pd.read_csv(args.reference_csv)
    validate_reference_table(reference_df, args.reference_csv)

    val_transform = transforms.Compose(
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

    all_results = []
    for display_name, spec in MODEL_SPECS.items():
        checkpoint_path = Path(spec["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for {display_name}: {checkpoint_path}"
            )

        print(f"\nLoading {display_name}: {checkpoint_path}")
        model = get_binary_model(
            model_name=spec["model_name"],
            pretrained=False,
        )
        checkpoint_metadata = load_checkpoint(model, checkpoint_path, device)
        model = model.to(device)
        model.eval()

        learning_rate = checkpoint_metadata.get("learning_rate")
        if learning_rate is None:
            learning_rate = spec["learning_rate"]
        source_run_id = checkpoint_metadata.get("source_wandb_run_id")
        if source_run_id is None:
            source_run_id = spec["source_wandb_run_id"]

        match_count = 0
        with torch.inference_mode():
            for _, row in reference_df.iterrows():
                image_name = os.path.basename(
                    str(row["image_path"]).replace("\\", "/")
                )
                image_path = Path(args.image_dir) / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                input_tensor = val_transform(image).unsqueeze(0).to(device)

                logits = model(input_tensor)
                probabilities = F.softmax(logits, dim=1).squeeze(0).cpu()
                score_bad = float(probabilities[0])
                score_good = float(probabilities[1])
                predicted_class = int(score_good > score_bad)
                predicted_label = "Good" if predicted_class == 1 else "Bad"
                human_label = str(row["human_label"])
                is_match = int(human_label == predicted_label)
                match_count += is_match

                all_results.append(
                    {
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "human_label": human_label,
                        "model": display_name,
                        "model_name": spec["model_name"],
                        "predicted_label": predicted_label,
                        "score_good": round(score_good, 6),
                        "score_bad": round(score_bad, 6),
                        "match": is_match,
                        "checkpoint_path": checkpoint_path.as_posix(),
                        "learning_rate": learning_rate,
                        "source_wandb_run_id": source_run_id,
                        "best_epoch": checkpoint_metadata.get("best_epoch"),
                        "best_val_f1": checkpoint_metadata.get("best_val_f1"),
                    }
                )

        print(f"{display_name}: {match_count}/100 predictions match human labels")

    results_df = pd.DataFrame(all_results)
    if len(results_df) != 300:
        raise RuntimeError(f"Expected 300 result rows, found {len(results_df)}")
    counts = results_df.groupby("model").size().to_dict()
    if counts != {"ResNet18": 100, "ResNet50": 100, "VGG16": 100}:
        raise RuntimeError(f"Unexpected result counts: {counts}")
    if results_df.duplicated(["model", "image_id"]).any():
        raise RuntimeError("Duplicate model/image_id rows found in inference results")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} rows: {output_path}")
    print("Validation passed: three models x 100 unique reference images")


if __name__ == "__main__":
    main()
