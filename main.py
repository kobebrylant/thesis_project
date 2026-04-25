#!/usr/bin/env python3
"""Main entry point for video game sentiment analysis training."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from src.config import (
    PathConfig,
    TrainingConfig,
    DeviceConfig,
    PhiloxSeedGenerator,
    CLASSICAL_MODELS,
    TRANSFORMER_MODELS,
)
from src.data_loader import DataLoader
from src.checkpoint_manager import CheckpointManager
from src.trainer import Trainer
from src.statistical_analysis import StatisticalAnalyzer, ThesisValidator
from src.visualization import Visualizer


def print_banner():
    print("=" * 70)
    print("VIDEO GAME SENTIMENT ANALYSIS")
    print("=" * 70)


def print_device_info():
    info = DeviceConfig.get_device_info()
    print(f"\nDevice: {info['device']}")
    if "gpu_name" in info:
        print(f"GPU: {info['gpu_name']}")
        print(f"Memory: {info['gpu_memory_gb']}")


def load_incomplete_models_info(path_config: PathConfig) -> Optional[Dict[str, str]]:
    metadata_path = path_config.checkpoints_dir / "electra_mocking_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        incomplete_models = {}
        model = metadata.get("model", "ELECTRA")
        real_runs = metadata.get("real_runs", 0)
        synthetic_runs = metadata.get("synthetic_runs", 0)

        if synthetic_runs > 0:
            incomplete_models[model] = (
                f"{real_runs} real runs, {synthetic_runs} mean-imputed"
            )

        return incomplete_models if incomplete_models else None

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load incomplete models metadata: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train sentiment analysis models with checkpoint support"
    )

    # --- Data gathering (granular) ---
    gather_group = parser.add_argument_group("Data gathering")
    gather_group.add_argument(
        "--gather-success-metrics",
        action="store_true",
        help="Collect market success metrics from SteamSpy and Steam Store only",
    )
    gather_group.add_argument(
        "--force-recollect",
        action="store_true",
        help="Force re-collection of success metrics even if file exists",
    )

    # --- ETL ---
    etl_group = parser.add_argument_group("ETL")
    etl_group.add_argument(
        "--build-dataset",
        action="store_true",
        help="Run ETL pipeline to rebuild parquet datasets from raw CSVs",
    )

    # --- Training ---
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--train",
        action="store_true",
        help="Run model training (requires datasets to already exist)",
    )
    train_group.add_argument(
        "--n-seeds",
        type=int,
        default=12,
        help="Number of random seeds to use (default: 12)",
    )
    train_group.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    train_group.add_argument(
        "--classical-only",
        action="store_true",
        help="Train only classical ML models",
    )
    train_group.add_argument(
        "--transformer-only",
        action="store_true",
        help="Train only transformer models",
    )
    train_group.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to train (e.g., XGBoost LightGBM ELECTRA)",
    )
    train_group.add_argument(
        "--no-save-models",
        action="store_true",
        help="Don't save trained model weights",
    )
    train_group.add_argument(
        "--fresh",
        action="store_true",
        help="Clear all checkpoints and saved models before training (fresh start)",
    )
    train_group.add_argument(
        "--import-checkpoint",
        type=Path,
        default=None,
        help="Import results from external CSV file",
    )

    # --- Analysis ---
    analysis_group = parser.add_argument_group("Analysis")
    analysis_group.add_argument(
        "--analyze",
        "--analyze-only",
        action="store_true",
        dest="analyze",
        help="Run statistical analysis on existing training results",
    )
    analysis_group.add_argument(
        "--visualize",
        action="store_true",
        help="Generate all visualizations from existing results",
    )
    analysis_group.add_argument(
        "--data-distribution",
        action="store_true",
        help="Generate data distribution analysis visualizations",
    )
    analysis_group.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for MOE calculation (default: 0.95)",
    )
    analysis_group.add_argument(
        "--positive-ratio",
        type=float,
        default=None,
        help="Positive class ratio for MCEN calculation",
    )
    analysis_group.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        help="Total dataset size for MCEN calculation",
    )
    analysis_group.add_argument(
        "--incomplete-model-handling",
        type=str,
        default="drop_incomplete_models",
        choices=["drop_incomplete_models", "drop_runs", "use_available"],
        help=(
            "How to handle models with incomplete results in statistical tests. "
            "'drop_incomplete_models' (default): exclude models with <50%% runs. "
            "'drop_runs': drop any run where any model has missing data. "
            "'use_available': keep rows with at least 50%% non-NaN values."
        ),
    )

    # --- Validation ---
    validation_group = parser.add_argument_group("Thesis validation")
    validation_group.add_argument(
        "--validate-thesis",
        action="store_true",
        help="Run thesis validation on UNSEEN validation games only (no data leakage)",
    )
    validation_group.add_argument(
        "--validation-model",
        type=str,
        default=None,
        help=(
            "Model to use for thesis validation "
            "(e.g. ELECTRA, RoBERTa, LogisticRegression, LightGBM). "
            "Default: first available, preferring transformers."
        ),
    )
    validation_group.add_argument(
        "--predict-success",
        action="store_true",
        help=(
            "Run LOO-CV success prediction (tier classification + revenue regression) "
            "with a Steam-only vs YouTube-only vs combined feature ablation."
        ),
    )
    validation_group.add_argument(
        "--youtube-sentiment-csv",
        type=Path,
        default=None,
        help=(
            "Path to YouTube sentiment features CSV "
            "(default: results/metrics/youtube_sentiment.csv)."
        ),
    )

    # --- Utility ---
    util_group = parser.add_argument_group("Utility")
    util_group.add_argument(
        "--status",
        action="store_true",
        help="Show training progress and exit",
    )
    util_group.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: ./data)",
    )
    util_group.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results directory (default: ./results)",
    )

    args = parser.parse_args()

    print_banner()
    print_device_info()

    base_dir = Path.cwd()
    path_config = PathConfig(
        base_dir=base_dir,
        data_dir=args.data_dir or base_dir / "data",
        results_dir=args.results_dir or base_dir / "results",
    )

    if args.gather_success_metrics:
        from success_data_gatherer import (
            SuccessMetricsCollector,
            EARLY_ACCESS_GAMES,
            VALIDATION_GAMES,
            OUTPUT_FILE,
            VALIDATION_OUTPUT_FILE,
        )

        print("\nCollecting market success metrics...")

        if args.force_recollect:
            for f in [OUTPUT_FILE, VALIDATION_OUTPUT_FILE]:
                if f.exists():
                    print(f"Removing existing file: {f}")
                    f.unlink()

        collector = SuccessMetricsCollector()

        print("\n--- Training games ---")
        results = collector.collect_all(
            EARLY_ACCESS_GAMES, resume=not args.force_recollect, output_file=OUTPUT_FILE
        )
        complete = sum(1 for r in results if r.data_complete)
        print(f"Training: {complete}/{len(results)} games with full data")

        print("\n--- Validation games ---")
        val_results = collector.collect_all(
            VALIDATION_GAMES,
            resume=not args.force_recollect,
            output_file=VALIDATION_OUTPUT_FILE,
        )
        complete = sum(1 for r in val_results if r.data_complete)
        print(f"Validation: {complete}/{len(val_results)} games with full data")
        return

    if args.build_dataset:
        import subprocess

        print("\nRebuilding parquet datasets...")
        result = subprocess.run(
            [sys.executable, "-m", "src.data_pipeline", "--force"],
            capture_output=False,
        )
        if result.returncode != 0:
            print("ERROR: ETL pipeline failed")
            sys.exit(1)
        return

    # --- Training setup ---
    training_config = TrainingConfig(
        n_seeds=args.n_seeds,
        n_folds=args.n_folds,
    )

    seed_generator = PhiloxSeedGenerator()
    seeds = seed_generator.generate_seeds(args.n_seeds)
    print(f"\nGenerated {len(seeds)} seeds using Philox RNG: {seeds[:12]}")

    checkpoint = CheckpointManager(
        path_config=path_config,
        n_seeds=args.n_seeds,
        n_folds=args.n_folds,
    )

    if args.import_checkpoint:
        checkpoint.import_from_csv(args.import_checkpoint)

    if args.status:
        checkpoint.print_progress(seeds)
        return

    if args.fresh:
        print("\n*** FRESH START - Clearing previous results ***")
        checkpoints_dir = path_config.checkpoints_dir
        models_dir = path_config.models_dir

        checkpoint_files = list(checkpoints_dir.glob("*.csv")) + list(
            checkpoints_dir.glob("*.json")
        )
        for f in checkpoint_files:
            f.unlink()
            print(f"  Removed: {f.name}")

        model_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pt"))
        for f in model_files:
            f.unlink()
        if model_files:
            print(f"  Removed: {len(model_files)} saved model files")

        print("*** Ready for fresh training ***\n")

        checkpoint = CheckpointManager(
            path_config=path_config,
            n_seeds=args.n_seeds,
            n_folds=args.n_folds,
        )

    if args.analyze:
        results_df = checkpoint.get_results_df()
        if results_df.empty:
            print("ERROR: No training results found. Run --train first.")
            sys.exit(1)

        analyzer = StatisticalAnalyzer(path_config)
        analyzer.run_full_analysis(
            results_df,
            n_seeds=args.n_seeds,
            n_folds=args.n_folds,
            confidence_level=args.confidence_level,
            incomplete_model_handling=args.incomplete_model_handling,
        )
        return

    if args.visualize:
        results_df = checkpoint.get_results_df()
        if results_df.empty:
            print("ERROR: No training results found. Run --train first.")
            sys.exit(1)

        visualizer = Visualizer(path_config)
        analyzer = StatisticalAnalyzer(path_config)

        analysis = analyzer.run_full_analysis(
            results_df,
            save_results=True,
            n_seeds=args.n_seeds,
            n_folds=args.n_folds,
            confidence_level=args.confidence_level,
            incomplete_model_handling=args.incomplete_model_handling,
        )

        summary_df = analysis.get("summary")
        if summary_df is None or summary_df.empty:
            summary_df = (
                results_df.groupby(["model", "type"])
                .agg(
                    {
                        "f1_score": ["mean", "std"],
                        "accuracy": ["mean", "std"],
                        "precision": ["mean", "std"],
                        "recall": ["mean", "std"],
                        "train_time": ["mean", "std"],
                        "inference_time_ms": ["mean", "std"],
                    }
                )
                .round(4)
            )
            summary_df.columns = ["_".join(col) for col in summary_df.columns]
            summary_df = summary_df.reset_index()

        incomplete_models = load_incomplete_models_info(path_config)

        visualizer.generate_all_visualizations(
            results_df=results_df,
            summary_df=summary_df,
            nemenyi_df=analysis.get("nemenyi"),
            wilcoxon_df=analysis.get("wilcoxon"),
            n_seeds=args.n_seeds,
            n_folds=args.n_folds,
            incomplete_models=incomplete_models,
        )
        return

    if args.data_distribution:
        print("\nLoading data for distribution analysis...")
        data_loader = DataLoader(path_config)
        try:
            data = data_loader.load_all_reviews()
            visualizer = Visualizer(path_config)
            visualizer.analyze_data_distribution(data.df)
            visualizer.plot_early_access_vs_post_release(data.df)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        return

    # --- Success prediction with multi-source ablation ---
    if args.predict_success:
        import pandas as pd

        from src.analysis import SuccessPredictor

        metrics_csv = Path("data/game_success_metrics.csv")
        sentiment_csv = path_config.metrics_dir / "training_game_sentiment.csv"
        yt_csv = args.youtube_sentiment_csv or (
            path_config.metrics_dir / "youtube_sentiment.csv"
        )

        if not metrics_csv.exists():
            print(f"\nERROR: {metrics_csv} not found.")
            print("Run: python main.py --gather-success-metrics")
            sys.exit(1)
        if not yt_csv.exists():
            print(f"\nERROR: {yt_csv} not found.")
            print("Run: python youtube_sentiment.py")
            sys.exit(1)

        if sentiment_csv.exists():
            print(f"\nUsing cached training sentiment: {sentiment_csv}")
            game_sentiment = pd.read_csv(sentiment_csv)
        else:
            print(
                f"\nTraining-games sentiment CSV not found — computing it now "
                f"(will cache to {sentiment_csv})"
            )
            validator = ThesisValidator(path_config)
            predictor_model = validator.load_predictor(args.validation_model)
            print(f"Predicting EA sentiment with {predictor_model.name}...")

            data_loader = DataLoader(path_config)
            data = data_loader.load_all_reviews()
            game_sentiment = validator.calculate_game_sentiment(data.df, predictor_model)
            sentiment_csv.parent.mkdir(parents=True, exist_ok=True)
            game_sentiment.to_csv(sentiment_csv, index=False)
            print(f"Saved training sentiment to {sentiment_csv}")

        success_metrics = pd.read_csv(metrics_csv)
        youtube_sentiment = pd.read_csv(yt_csv)

        print("\n" + "=" * 70)
        print("SUCCESS PREDICTION — multi-source ablation (training games)")
        print("=" * 70)
        print(f"Steam sentiment:   {sentiment_csv} ({len(game_sentiment)} games)")
        print(f"Success metrics:   {metrics_csv} ({len(success_metrics)} games)")
        print(f"YouTube sentiment: {yt_csv} ({len(youtube_sentiment)} games)")

        success_predictor = SuccessPredictor(path_config)
        ablation_df = success_predictor.run_ablation(
            game_sentiment, success_metrics, youtube_sentiment
        )

        out_path = path_config.metrics_dir / "success_prediction_ablation.csv"
        ablation_df.to_csv(out_path, index=False)
        print(f"\nAblation results saved to {out_path}")
        return

    # --- Thesis validation (always uses validation dataset only) ---
    if args.validate_thesis:
        print("\n" + "=" * 70)
        print("THESIS VALIDATION ON UNSEEN GAMES")
        print("=" * 70)
        print("\nThis is the scientifically correct validation method.")
        print("The model will predict on games it has NEVER seen during training.\n")

        validator = ThesisValidator(path_config)
        try:
            validation_results = validator.run_validation_on_new_games(
                model_name=args.validation_model,
            )

            if "game_sentiment" in validation_results:
                visualizer = Visualizer(path_config)
                if "success_metrics" in validation_results:
                    visualizer.generate_validation_plots(
                        game_sentiment=validation_results["game_sentiment"],
                        success_metrics=validation_results["success_metrics"],
                        success_correlations=validation_results.get(
                            "success_correlations", {}
                        ),
                        post_release_correlation=validation_results.get(
                            "post_release_correlation"
                        ),
                    )
                else:
                    visualizer.plot_thesis_validation(
                        validation_results["game_sentiment"],
                        validation_results.get("post_release_correlation"),
                    )
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("\nTo collect validation data, run:")
            print("  python data_gatherer.py --validation")
            sys.exit(1)
        return

    has_action = any([
        args.gather_success_metrics,
        args.build_dataset,
        args.analyze,
        args.visualize,
        args.data_distribution,
        args.validate_thesis,
        args.predict_success,
        args.status,
    ])

    if not args.train and has_action:
        return

    print("\nLoading data...")
    data_loader = DataLoader(path_config)

    try:
        data = data_loader.load_all_reviews()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"Please ensure your data is in: {path_config.data_dir}")
        print("Run --build-dataset first if you haven't built parquet files.")
        sys.exit(1)

    print(f"\n{data.summary()}")

    train_classical = not args.transformer_only
    train_transformers = not args.classical_only

    if args.models:
        classical_models = [m for m in args.models if m in CLASSICAL_MODELS]
        transformer_models = [m for m in args.models if m in TRANSFORMER_MODELS]
        train_classical = len(classical_models) > 0
        train_transformers = len(transformer_models) > 0
    else:
        classical_models = None
        transformer_models = None

    trainer = Trainer(
        path_config=path_config,
        training_config=training_config,
        checkpoint_manager=checkpoint,
    )

    checkpoint.print_progress(seeds)

    save_models = not args.no_save_models

    if train_classical:
        trainer.train_classical_models(
            X=data.X,
            y=data.y,
            seeds=seeds,
            models=classical_models,
            save_models=save_models,
        )

    if train_transformers:
        trainer.train_transformer_models(
            X=data.X,
            y=data.y,
            seeds=seeds,
            models=transformer_models,
            save_models=save_models,
        )

    checkpoint.print_progress(seeds)

    results_df = checkpoint.get_results_df()
    if not results_df.empty and results_df["model"].nunique() >= 2:
        print("\nRunning statistical analysis...")
        analyzer = StatisticalAnalyzer(path_config)

        analyzer.run_full_analysis(
            results_df,
            n_seeds=args.n_seeds,
            n_folds=args.n_folds,
            confidence_level=args.confidence_level,
            incomplete_model_handling=args.incomplete_model_handling,
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
