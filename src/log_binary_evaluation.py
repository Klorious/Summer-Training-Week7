import argparse

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PREDICTIONS_CSV = "outputs/predictions/reference_results.csv"
METRICS_CSV = "outputs/predictions/binary_reference_metrics.csv"
REFERENCE_CSV = "data/annotations/binary_reference_100.csv"

EXPECTED_REFERENCE_SIZE = 100
EXPECTED_MODELS = ("VGG16", "ResNet18", "ResNet50")
LABEL_MAP = {"Bad": 0, "Good": 1}

EXPECTED_PROVENANCE = {
    "VGG16": {
        "model_name": "vgg16",
        "checkpoint_path": "outputs/checkpoints/best_vgg16_lr0.0001.pth",
        "learning_rate": 0.0001,
        "source_wandb_run_id": "sgxudag7",
        "best_epoch": 4,
        "best_val_f1": 0.8307692307692308,
    },
    "ResNet18": {
        "model_name": "resnet18",
        "checkpoint_path": "outputs/checkpoints/best_resnet18_lr0.001.pth",
        "learning_rate": 0.001,
        "source_wandb_run_id": "6d6u2kul",
        "best_epoch": 17,
        "best_val_f1": 0.8111888111888111,
    },
    "ResNet50": {
        "model_name": "resnet50",
        "checkpoint_path": "outputs/checkpoints/best_resnet50_lr0.0001.pth",
        "learning_rate": 0.0001,
        "source_wandb_run_id": "af8dkst0",
        "best_epoch": 5,
        "best_val_f1": 0.8652482269503546,
    },
}

VALIDATION_RUNS = [
    {
        "model": "ResNet18",
        "learning_rate": 0.0001,
        "best_epoch": 2,
        "best_val_f1": 0.7868852459016393,
        "wandb_run_id": "z9fy0ef9",
        "selected_for_reference_inference": False,
    },
    {
        "model": "ResNet18",
        "learning_rate": 0.001,
        "best_epoch": 17,
        "best_val_f1": 0.8111888111888111,
        "wandb_run_id": "6d6u2kul",
        "selected_for_reference_inference": True,
    },
    {
        "model": "VGG16",
        "learning_rate": 0.0001,
        "best_epoch": 4,
        "best_val_f1": 0.8307692307692308,
        "wandb_run_id": "sgxudag7",
        "selected_for_reference_inference": True,
    },
    {
        "model": "ResNet50",
        "learning_rate": 0.0001,
        "best_epoch": 5,
        "best_val_f1": 0.8652482269503546,
        "wandb_run_id": "af8dkst0",
        "selected_for_reference_inference": True,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Log the finalized Unit 7 binary evaluation to W&B."
    )
    parser.add_argument("--entity", default="klorius-")
    parser.add_argument("--project", default="training-Unit7-Binary")
    parser.add_argument(
        "--run-name",
        default="binary_final_evaluation_seed20260725",
    )
    parser.add_argument(
        "--run-id",
        default="bkfn0726",
        help="Fixed W&B run ID prevents accidental duplicate final runs.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def calculate_metrics(group):
    y_true = group["human_label"].map(LABEL_MAP)
    y_pred = group["predicted_label"].map(LABEL_MAP)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def load_and_validate_results():
    predictions = pd.read_csv(PREDICTIONS_CSV)
    stored_metrics = pd.read_csv(METRICS_CSV)
    reference = pd.read_csv(REFERENCE_CSV)

    required_prediction_columns = {
        "image_id",
        "image_path",
        "human_label",
        "model",
        "model_name",
        "predicted_label",
        "score_good",
        "score_bad",
        "match",
        "checkpoint_path",
        "learning_rate",
        "source_wandb_run_id",
        "best_epoch",
        "best_val_f1",
    }
    missing = required_prediction_columns.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"{PREDICTIONS_CSV} is missing columns: {', '.join(sorted(missing))}"
        )

    required_metric_columns = {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    missing = required_metric_columns.difference(stored_metrics.columns)
    if missing:
        raise ValueError(
            f"{METRICS_CSV} is missing columns: {', '.join(sorted(missing))}"
        )

    if len(reference) != EXPECTED_REFERENCE_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_REFERENCE_SIZE} reference rows, found {len(reference)}"
        )
    if reference["image_id"].isna().any() or reference["image_id"].duplicated().any():
        raise ValueError("Reference CSV has missing or duplicate image IDs")
    if set(reference["human_label"]) != {"Bad", "Good"}:
        raise ValueError("Reference CSV must contain both Bad and Good labels")

    reference_ids = set(reference["image_id"].astype(str))
    if len(predictions) != EXPECTED_REFERENCE_SIZE * len(EXPECTED_MODELS):
        raise ValueError(
            f"Expected 300 prediction rows, found {len(predictions)}"
        )
    if set(predictions["model"]) != set(EXPECTED_MODELS):
        raise ValueError("Prediction CSV does not contain the expected three models")
    if predictions.duplicated(["model", "image_id"]).any():
        raise ValueError("Prediction CSV has duplicate model/image_id rows")
    if not predictions["human_label"].isin(LABEL_MAP).all():
        raise ValueError("Prediction CSV contains invalid human labels")
    if not predictions["predicted_label"].isin(LABEL_MAP).all():
        raise ValueError("Prediction CSV contains invalid predicted labels")

    numeric_columns = [
        "score_good",
        "score_bad",
        "learning_rate",
        "best_epoch",
        "best_val_f1",
    ]
    numeric = predictions[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Prediction CSV has missing or non-numeric values")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Prediction CSV has infinite numeric values")
    if not np.allclose(
        predictions["score_good"] + predictions["score_bad"],
        1.0,
        atol=2e-6,
    ):
        raise ValueError("Bad/Good probabilities do not sum to one")

    reference_labels = reference.set_index("image_id")["human_label"].astype(str)
    recomputed_rows = []
    confusion_data = {}

    for model in EXPECTED_MODELS:
        group = predictions.loc[predictions["model"] == model].copy()
        if len(group) != EXPECTED_REFERENCE_SIZE:
            raise ValueError(
                f"{model} should have {EXPECTED_REFERENCE_SIZE} rows, found {len(group)}"
            )
        if set(group["image_id"].astype(str)) != reference_ids:
            raise ValueError(f"{model} did not evaluate the exact reference set")

        group_labels = group.set_index("image_id")["human_label"].astype(str)
        if not group_labels.sort_index().equals(reference_labels.sort_index()):
            raise ValueError(f"{model} human labels do not match the reference CSV")

        expected = EXPECTED_PROVENANCE[model]
        for field in (
            "model_name",
            "checkpoint_path",
            "source_wandb_run_id",
            "best_epoch",
        ):
            values = group[field].drop_duplicates().tolist()
            if values != [expected[field]]:
                raise ValueError(
                    f"{model} has unexpected {field}: {values}; "
                    f"expected {expected[field]}"
                )
        for field in ("learning_rate", "best_val_f1"):
            values = pd.to_numeric(group[field]).drop_duplicates().to_numpy()
            if len(values) != 1 or not np.isclose(
                float(values[0]), float(expected[field]), atol=1e-9
            ):
                raise ValueError(f"{model} has unexpected {field}: {values}")

        metrics = calculate_metrics(group)
        recomputed_rows.append({"model": model, **metrics})
        confusion_data[model] = {
            "y_true": group["human_label"].map(LABEL_MAP).tolist(),
            "y_pred": group["predicted_label"].map(LABEL_MAP).tolist(),
        }

    recomputed_metrics = pd.DataFrame(recomputed_rows)
    stored = stored_metrics.set_index("model").sort_index()
    recomputed = recomputed_metrics.set_index("model").sort_index()
    if set(stored.index) != set(recomputed.index):
        raise ValueError("Stored metric models do not match recomputed metrics")

    for column in ("accuracy", "precision", "recall", "f1"):
        if not np.allclose(stored[column], recomputed[column], atol=1e-12):
            raise ValueError(f"Stored {column} values do not match recomputation")
    for column in ("tn", "fp", "fn", "tp"):
        if not np.array_equal(
            stored[column].astype(int),
            recomputed[column].astype(int),
        ):
            raise ValueError(f"Stored {column} values do not match recomputation")

    return predictions, recomputed_metrics, confusion_data


def main():
    args = parse_args()
    predictions, metrics, confusion_data = load_and_validate_results()
    validation_runs = pd.DataFrame(VALIDATION_RUNS)

    reference_counts = (
        predictions.loc[predictions["model"] == EXPECTED_MODELS[0], "human_label"]
        .value_counts()
        .to_dict()
    )
    config = {
        "task": "binary_classification_final_evaluation",
        "evaluation_protocol": "locked held-out human reference set",
        "reference_size": EXPECTED_REFERENCE_SIZE,
        "reference_bad_count": int(reference_counts.get("Bad", 0)),
        "reference_good_count": int(reference_counts.get("Good", 0)),
        "label_encoding": "Bad=0, Good=1",
        "checkpoint_selection_metric": "validation F1",
        "reported_metrics": [
            "accuracy",
            "precision",
            "recall",
            "F1",
            "confusion matrix",
        ],
        "random_seed": 20260725,
        "framework": "PyTorch 2.6 reproducible reruns",
        "human_reference_used_for_training": False,
        "human_reference_used_for_validation": False,
        "human_reference_used_for_checkpoint_selection": False,
        "human_reference_used_for_hyperparameter_selection": False,
        "human_reference_usage": "final comparison only",
        "formal_training_run_ids": [
            "z9fy0ef9",
            "6d6u2kul",
            "sgxudag7",
            "af8dkst0",
        ],
        "selected_resnet18_learning_rate": 0.001,
        "selected_resnet18_run_id": "6d6u2kul",
        "best_validation_architecture": "ResNet50",
        "best_validation_f1": 0.8652482269503546,
    }

    with wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        id=args.run_id,
        resume="never",
        job_type="evaluation",
        tags=["basic", "binary", "final-evaluation", "report"],
        notes=(
            "Final held-out reference evaluation after the architecture, "
            "ResNet18 learning rate, epochs, and checkpoints were selected "
            "using validation F1 only."
        ),
        config=config,
        mode=args.wandb_mode,
    ) as run:
        run.log(
            {
                "reference_predictions": wandb.Table(dataframe=predictions),
                "binary_reference_metrics": wandb.Table(dataframe=metrics),
                "validation_run_comparison": wandb.Table(
                    dataframe=validation_runs
                ),
            }
        )

        for model in EXPECTED_MODELS:
            data = confusion_data[model]
            run.log(
                {
                    f"{model}/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=data["y_true"],
                        preds=data["y_pred"],
                        class_names=["Bad", "Good"],
                    )
                }
            )

        for _, row in metrics.iterrows():
            model = row["model"]
            for metric in ("accuracy", "precision", "recall", "f1"):
                run.summary[f"{model}/{metric}"] = float(row[metric])
            for count in ("tn", "fp", "fn", "tp"):
                run.summary[f"{model}/{count}"] = int(row[count])

        best_reference_row = metrics.loc[metrics["f1"].idxmax()]
        run.summary["best_reference_f1_model"] = best_reference_row["model"]
        run.summary["best_reference_f1"] = float(best_reference_row["f1"])
        run.summary["best_validation_model"] = "ResNet50"
        run.summary["best_validation_f1"] = 0.8652482269503546
        run.summary["selected_resnet18_learning_rate"] = 0.001
        run.summary["validation_status"] = (
            "300 rows; three models x 100 unique reference images; "
            "metrics and provenance independently validated"
        )

        artifact = wandb.Artifact(
            name="unit7-binary-final-results",
            type="evaluation",
            description=(
                "Final three-model reference predictions and independently "
                "validated binary classification metrics."
            ),
        )
        artifact.add_file(PREDICTIONS_CSV)
        artifact.add_file(METRICS_CSV)
        run.log_artifact(artifact)

        print(f"W&B run URL: {run.url}")
        print(
            f"Best held-out reference F1: {best_reference_row['model']} "
            f"({float(best_reference_row['f1']):.4f})"
        )
        print("Best validation F1: ResNet50 (0.8652)")
        print(
            "Validation passed: 300 predictions, three exact reference sets, "
            "finite probabilities, matching metrics, and locked provenance"
        )


if __name__ == "__main__":
    main()
