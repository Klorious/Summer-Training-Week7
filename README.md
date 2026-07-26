# Training Unit 7: Image Quality Classification and Ranking

This repository contains the reproducible code, data splits, evaluation
outputs, and Overleaf report source for:

1. Good/Bad binary classification with VGG16, ResNet18, and ResNet50.
2. Continuous image-quality regression and Top-100 ranking evaluation.

The 500 source images and model checkpoints are intentionally excluded from
Git. See `ENVIRONMENT.md` for the final Python, PyTorch, CUDA, and dependency
information.

## Data protocol

- Basic task: 300 training, 100 validation, and 100 locked reference images.
- Advanced task: a fixed 500-image candidate pool, 100 locked human Top-100
  images, 300 ranking-training images, and 100 ranking-validation images.
- The locked human reference sets are used only for final evaluation.

## Main commands

Train the basic binary classifiers:

```bash
python src/train_binary.py --config configs/vgg16.yaml
python src/train_binary.py --config configs/resnet18.yaml
python src/train_binary.py --config configs/resnet50.yaml
```

Evaluate the locked 100-image binary reference set:

```bash
python src/infer.py
python src/log_binary_evaluation.py
```

Run the three selected binary classifiers on all 500 candidate images:

```bash
python src/infer_candidate_pool.py
```

This produces:

```text
outputs/predictions/binary_candidate_pool_predictions.csv
outputs/predictions/binary_candidate_pool_distribution.csv
```

The prediction file must contain 1,500 rows: three models times 500 unique
candidate images.

Train and evaluate the advanced ranking models:

```bash
python src/train_ranking.py --config configs/ranking_vgg16.yaml
python src/train_ranking.py --config configs/ranking_resnet18.yaml
python src/train_ranking.py --config configs/ranking_resnet50.yaml
python src/score_candidate_pool.py
python src/evaluate_ranking.py
python src/log_ranking_evaluation.py
```

## Final results

The main machine-readable results are stored in:

```text
outputs/predictions/binary_reference_metrics.csv
outputs/predictions/reference_results.csv
outputs/ranking/all_model_scores.csv
outputs/ranking/ranking_metrics.csv
outputs/ranking/*_top100.csv
```

The Overleaf source is in `report/`. A ready-to-upload archive is generated at
`outputs/overleaf/Training-Unit7-Report-Overleaf.zip`.

## Experiment links

- GitHub: https://github.com/Klorious/Summer-Training-Week7
- Basic W&B: https://wandb.ai/klorius-/training-Unit7-Binary
- Advanced W&B: https://wandb.ai/klorius-/training-Unit7-Ranking
