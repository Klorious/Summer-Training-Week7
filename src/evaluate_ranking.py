import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


SCORES_CSV = "outputs/ranking/all_model_scores.csv"
REFERENCE_CSV = "data/annotations/ranking_reference_top100.csv"
OUTPUT_DIR = "outputs/ranking"

EXPECTED_POOL_SIZE = 500
EXPECTED_REFERENCE_SIZE = 100
TOP_K = 100

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


def validate_inputs(score_frame, reference_frame):
    required_score_columns = {"image_id", "image_path", *SCORE_COLUMNS}
    missing_score_columns = required_score_columns.difference(score_frame.columns)
    if missing_score_columns:
        missing = ", ".join(sorted(missing_score_columns))
        raise ValueError(f"{SCORES_CSV} is missing columns: {missing}")

    if len(score_frame) != EXPECTED_POOL_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_POOL_SIZE} scored images, "
            f"but found {len(score_frame)}"
        )
    if score_frame["image_id"].isna().any():
        raise ValueError(f"{SCORES_CSV} contains missing image_id values")
    if score_frame["image_id"].duplicated().any():
        raise ValueError(f"{SCORES_CSV} contains duplicate image_id values")

    numeric_scores = score_frame[SCORE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric_scores.isna().any().any():
        bad_columns = numeric_scores.columns[numeric_scores.isna().any()].tolist()
        raise ValueError(f"NaN or non-numeric scores found in: {bad_columns}")
    if not np.isfinite(numeric_scores.to_numpy(dtype=np.float64)).all():
        raise ValueError("Infinite values found in ranking scores")

    if "image_id" not in reference_frame.columns:
        raise ValueError(f"{REFERENCE_CSV} is missing the image_id column")
    if len(reference_frame) != EXPECTED_REFERENCE_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_REFERENCE_SIZE} human reference images, "
            f"but found {len(reference_frame)}"
        )
    if reference_frame["image_id"].isna().any():
        raise ValueError(f"{REFERENCE_CSV} contains missing image_id values")
    if reference_frame["image_id"].duplicated().any():
        raise ValueError(f"{REFERENCE_CSV} contains duplicate image_id values")

    score_ids = set(score_frame["image_id"].astype(str))
    reference_ids = set(reference_frame["image_id"].astype(str))
    missing_reference_ids = reference_ids.difference(score_ids)
    if missing_reference_ids:
        examples = sorted(missing_reference_ids)[:10]
        raise ValueError(
            "Human reference contains images missing from the candidate scores: "
            f"{examples}"
        )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    score_frame = pd.read_csv(SCORES_CSV)
    reference_frame = pd.read_csv(REFERENCE_CSV)
    validate_inputs(score_frame, reference_frame)

    reference_ids = set(reference_frame["image_id"].astype(str))
    score_frame["is_human_top100"] = (
        score_frame["image_id"].astype(str).isin(reference_ids).astype(int)
    )

    metrics_records = []

    for score_column in SCORE_COLUMNS:
        print(f"Evaluating: {score_column}")

        # A fixed image_id tie-break makes Top-100 selection reproducible if
        # two images receive exactly the same score.
        sorted_frame = score_frame.sort_values(
            by=[score_column, "image_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        top100_frame = sorted_frame.head(TOP_K).copy()
        if len(top100_frame) != TOP_K:
            raise ValueError(
                f"{score_column} produced only {len(top100_frame)} Top-K rows"
            )
        if top100_frame["image_id"].duplicated().any():
            raise ValueError(f"{score_column} Top-100 contains duplicate images")

        top100_output_path = os.path.join(
            OUTPUT_DIR, f"{score_column}_top100.csv"
        )
        top100_frame[["image_id", "image_path", score_column]].to_csv(
            top100_output_path, index=False
        )

        model_top100_ids = set(top100_frame["image_id"].astype(str))
        intersection = len(reference_ids.intersection(model_top100_ids))
        overlap = intersection / TOP_K
        union_size = len(reference_ids.union(model_top100_ids))
        jaccard = intersection / union_size

        # AP uses all 500 continuous scores and one binary relevance vector
        # derived from the single locked human Top-100 reference set.
        ap = average_precision_score(
            score_frame["is_human_top100"],
            score_frame[score_column],
        )

        metrics_records.append(
            {
                "ranking_method": score_column,
                "intersection": int(intersection),
                "overlap": round(float(overlap), 4),
                "jaccard": round(float(jaccard), 4),
                "ap": round(float(ap), 4),
            }
        )

    metrics_frame = pd.DataFrame(metrics_records)
    metrics_output_path = os.path.join(OUTPUT_DIR, "ranking_metrics.csv")
    metrics_frame.to_csv(metrics_output_path, index=False)

    print(f"\nSaved ranking metrics: {metrics_output_path}")
    print(metrics_frame.to_string(index=False))
    print(
        "\nValidation passed: eight unique Top-100 tables were produced; "
        "AP used continuous scores over all 500 candidate images"
    )


if __name__ == "__main__":
    main()
