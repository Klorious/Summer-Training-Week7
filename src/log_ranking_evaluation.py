import argparse
import os

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import average_precision_score


SCORES_CSV = "outputs/ranking/all_model_scores.csv"
METRICS_CSV = "outputs/ranking/ranking_metrics.csv"
REFERENCE_CSV = "data/annotations/ranking_reference_top100.csv"
OUTPUT_DIR = "outputs/ranking"

EXPECTED_POOL_SIZE = 500
EXPECTED_REFERENCE_SIZE = 100
TOP_K = 100
RANDOM_SEED = 20260725
TTA_RUNS = 5
PENALTY_WEIGHT = 0.5

SCORE_COLUMNS = [
    "vgg16_raw_score",
    "vgg16_adjusted_score",
    "resnet18_raw_score",
    "resnet18_adjusted_score",
    "resnet50_raw_score",
    "resnet50_adjusted_score",
    "ensemble_raw_score",
    "ensemble_adjusted_score",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Log the finalized Unit 7 ranking evaluation to W&B."
    )
    parser.add_argument("--entity", default="klorius-")
    parser.add_argument("--project", default="training-Unit7-Ranking")
    parser.add_argument(
        "--run-name",
        default="ranking_final_evaluation_seed20260725",
    )
    parser.add_argument(
        "--run-id",
        default="rkfn0726",
        help="Fixed W&B run ID prevents accidental duplicate final runs.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def load_and_validate_results():
    scores = pd.read_csv(SCORES_CSV)
    metrics = pd.read_csv(METRICS_CSV)
    reference = pd.read_csv(REFERENCE_CSV)

    required_score_columns = {"image_id", "image_path", *SCORE_COLUMNS}
    missing_score_columns = required_score_columns.difference(scores.columns)
    if missing_score_columns:
        missing = ", ".join(sorted(missing_score_columns))
        raise ValueError(f"{SCORES_CSV} is missing columns: {missing}")

    if len(scores) != EXPECTED_POOL_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_POOL_SIZE} score rows, found {len(scores)}"
        )
    if scores["image_id"].isna().any() or scores["image_id"].duplicated().any():
        raise ValueError(f"{SCORES_CSV} has missing or duplicate image_id values")

    numeric_scores = scores[SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric_scores.isna().any().any():
        raise ValueError(f"{SCORES_CSV} has NaN or non-numeric scores")
    if not np.isfinite(numeric_scores.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{SCORES_CSV} has infinite scores")

    if "image_id" not in reference.columns:
        raise ValueError(f"{REFERENCE_CSV} is missing image_id")
    if len(reference) != EXPECTED_REFERENCE_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_REFERENCE_SIZE} reference rows, "
            f"found {len(reference)}"
        )
    if reference["image_id"].isna().any() or reference["image_id"].duplicated().any():
        raise ValueError(
            f"{REFERENCE_CSV} has missing or duplicate image_id values"
        )

    required_metric_columns = {
        "ranking_method",
        "intersection",
        "overlap",
        "jaccard",
        "ap",
    }
    missing_metric_columns = required_metric_columns.difference(metrics.columns)
    if missing_metric_columns:
        missing = ", ".join(sorted(missing_metric_columns))
        raise ValueError(f"{METRICS_CSV} is missing columns: {missing}")
    if len(metrics) != len(SCORE_COLUMNS):
        raise ValueError(
            f"Expected {len(SCORE_COLUMNS)} metric rows, found {len(metrics)}"
        )
    if set(metrics["ranking_method"]) != set(SCORE_COLUMNS):
        raise ValueError("Ranking methods in metrics CSV do not match score columns")

    reference_ids = set(reference["image_id"].astype(str))
    score_ids = set(scores["image_id"].astype(str))
    if not reference_ids.issubset(score_ids):
        raise ValueError("Human Top-100 is not a subset of the scored candidate pool")

    relevance = scores["image_id"].astype(str).isin(reference_ids).astype(int)
    metrics_by_method = metrics.set_index("ranking_method")
    top100_tables = {}

    for score_column in SCORE_COLUMNS:
        top100_path = os.path.join(
            OUTPUT_DIR, f"{score_column}_top100.csv"
        )
        top100 = pd.read_csv(top100_path)
        if len(top100) != TOP_K:
            raise ValueError(
                f"{top100_path} should have {TOP_K} rows, found {len(top100)}"
            )
        if top100["image_id"].isna().any() or top100["image_id"].duplicated().any():
            raise ValueError(
                f"{top100_path} has missing or duplicate image_id values"
            )

        top100_ids = set(top100["image_id"].astype(str))
        intersection = len(reference_ids.intersection(top100_ids))
        overlap = intersection / TOP_K
        jaccard = intersection / len(reference_ids.union(top100_ids))
        ap = average_precision_score(relevance, scores[score_column])

        stored = metrics_by_method.loc[score_column]
        if int(stored["intersection"]) != intersection:
            raise ValueError(f"{score_column} intersection does not match")
        if not np.isclose(float(stored["overlap"]), overlap, atol=5e-5):
            raise ValueError(f"{score_column} overlap does not match")
        if not np.isclose(float(stored["jaccard"]), jaccard, atol=5e-5):
            raise ValueError(f"{score_column} Jaccard does not match")
        if not np.isclose(float(stored["ap"]), ap, atol=5e-5):
            raise ValueError(f"{score_column} AP does not match")

        top100_tables[score_column] = top100

    return scores, metrics, top100_tables


def main():
    args = parse_args()
    scores, metrics, top100_tables = load_and_validate_results()

    config = {
        "evaluation_protocol": "locked human Top-100 final comparison",
        "candidate_pool_size": EXPECTED_POOL_SIZE,
        "human_reference_size": EXPECTED_REFERENCE_SIZE,
        "top_k": TOP_K,
        "random_seed": RANDOM_SEED,
        "tta_runs": TTA_RUNS,
        "confidence_penalty_weight": PENALTY_WEIGHT,
        "confidence_adjusted_formula": "TTA mean - 0.5 * TTA standard deviation",
        "ensemble_formula": "mean of per-model z-scores",
        "output_format": "continuous_regression_scalar",
        "quality_target_scale": "1-10 human quality score",
        "observed_training_range": "4-9",
        "prediction_output_range": "unbounded real-valued regression output",
        "higher_is_better": True,
        "human_reference_used_for_training": False,
        "human_reference_used_for_validation": False,
        "human_reference_used_for_checkpoint_selection": False,
        "human_reference_usage": "final comparison only",
        "ap_definition": "Average Precision over all 500 continuous scores",
        "reported_metric_name": "AP (one human reference set; not mAP)",
        "ranking_methods": SCORE_COLUMNS,
    }

    with wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        id=args.run_id,
        resume="never",
        job_type="evaluation",
        tags=["advanced", "final-evaluation", "top100"],
        notes=(
            "Final evaluation after models, hyperparameters, human Top-100, "
            "TTA penalty, and ensemble definition were locked."
        ),
        config=config,
        mode=args.wandb_mode,
    ) as run:
        tables = {
            "full_candidate_scores": wandb.Table(dataframe=scores),
            "ranking_metrics": wandb.Table(dataframe=metrics),
        }
        for score_column, top100 in top100_tables.items():
            tables[f"top100_{score_column}"] = wandb.Table(dataframe=top100)
        run.log(tables)

        for _, row in metrics.iterrows():
            method = row["ranking_method"]
            run.summary[f"{method}/intersection"] = int(row["intersection"])
            run.summary[f"{method}/overlap"] = float(row["overlap"])
            run.summary[f"{method}/jaccard"] = float(row["jaccard"])
            run.summary[f"{method}/ap"] = float(row["ap"])

        best_ap_row = metrics.loc[metrics["ap"].idxmax()]
        best_overlap_row = metrics.loc[metrics["overlap"].idxmax()]
        run.summary["best_ap_method"] = best_ap_row["ranking_method"]
        run.summary["best_ap"] = float(best_ap_row["ap"])
        run.summary["best_overlap_method"] = best_overlap_row["ranking_method"]
        run.summary["best_overlap"] = float(best_overlap_row["overlap"])
        run.summary["validation_status"] = (
            "500 unique finite score rows; eight valid Top-100 tables"
        )

        artifact = wandb.Artifact(
            name="unit7-ranking-final-results",
            type="evaluation",
            description=(
                "Final deterministic candidate scores, eight Top-100 tables, "
                "and ranking metrics."
            ),
        )
        artifact.add_file(SCORES_CSV)
        artifact.add_file(METRICS_CSV)
        for score_column in SCORE_COLUMNS:
            artifact.add_file(
                os.path.join(OUTPUT_DIR, f"{score_column}_top100.csv")
            )
        run.log_artifact(artifact)

        print(f"W&B run URL: {run.url}")
        print(
            f"Best AP: {best_ap_row['ranking_method']} "
            f"({float(best_ap_row['ap']):.4f})"
        )
        print(
            f"Best Overlap: {best_overlap_row['ranking_method']} "
            f"({float(best_overlap_row['overlap']):.4f})"
        )


if __name__ == "__main__":
    main()
