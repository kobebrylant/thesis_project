# Predicting Video Game Success through Pre-Release Sentiment Analysis during Early Access

Master's thesis project

The system trains sentiment classifiers on Steam Early Access reviews, then uses predicted sentiment scores as features to predict commercial success (revenue tier) for unreleased games.

## Overview

1. Collect Steam reviews and metadata for Early Access games
2. Train sentiment classifiers (classical ML and transformer models) using stratified k-fold cross-validation
3. Apply classifiers to pre-release review windows to generate per-game sentiment features
4. Predict commercial success tier using leave-one-out cross-validation on the small validation set

## Requirements

```bash
pip install -r requirements.txt
```

For YouTube comment collection, install the Google API client and set your key:

```bash
pip install google-api-python-client
export YOUTUBE_API_KEY=your_key_here
```

## Data Collection

All data collection is done via `data_gatherer.py`. Steam reviews and metadata are collected together.

```bash
# Collect Steam reviews and metadata
python data_gatherer.py --training        # training games only
python data_gatherer.py --validation      # validation games only
python data_gatherer.py --all             # both datasets

# Include YouTube trailer comments
python data_gatherer.py --all --youtube

# Force re-collection (ignore existing files)
python data_gatherer.py --all --fresh
```

Success metrics (SteamSpy owner estimates and Steam Store pricing) are collected separately:

```bash
python main.py --gather-success-metrics
python main.py --gather-success-metrics --force-recollect
```

All raw data is saved to `data/` (training) and `data/validation/` (validation).

## ETL Pipeline

Converts raw CSVs to Parquet datasets used during training. The pipeline hashes source files and skips rebuilds unless data has changed.

```bash
python main.py --build-dataset
python -m src.data_pipeline --force    # force rebuild
python -m src.data_pipeline --status   # check staleness
```

## Training

```bash
python main.py                         # default: train all models, 12 seeds, 5 folds
python main.py --train --n-seeds 12
python main.py --fresh --n-seeds 12    # clear checkpoints and restart
python main.py --classical-only
python main.py --transformer-only
python main.py --models XGBoost LightGBM
python main.py --status                # show checkpoint progress
```

Training resumes automatically from checkpoints. Completed `(model, seed, fold)` triples are skipped on re-run.

## Models

| Type        | Models                                                                   | Input                             |
| ----------- | ------------------------------------------------------------------------ | --------------------------------- |
| Classical   | XGBoost, LightGBM, SVM, LogisticRegression, NaiveBayes                   | TF-IDF (10k features, 1-2 ngrams) |
| Transformer | ELECTRA (`google/electra-small-discriminator`), RoBERTa (`roberta-base`) | Raw text, max 128 tokens          |

TF-IDF is fit per fold to prevent data leakage.

## Analysis and Visualisation

```bash
python main.py --analyze               # statistical analysis on training results
python main.py --visualize             # generate figures
python main.py --data-distribution     # data distribution plots
```

## Thesis Validation

Runs the full success prediction pipeline on held-out validation games only (never seen during training).

```bash
python main.py --validate-thesis
```

Revenue is estimated using the Boxleiter method: `total_reviews x 30 x current_price_usd`. Success tier prediction uses leave-one-out cross-validation.

## Statistical Verification

Cross-checks results against PSPP output for statistical correctness.

```bash
python -m test_utils.statistical_verification --all
```

## Configuration

Key parameters are in `src/config.py`:

- `TrainingConfig`: `n_seeds=12`, `n_folds=5`, `tfidf_max_features=10000`, `transformer_max_length=128`
- `CLASSICAL_PARAMS`: per-model hyperparameter dicts
- `TRANSFORMER_PATHS`: HuggingFace model identifiers
- `DeviceConfig.get_device()`: auto-selects CUDA, then MPS, then CPU
