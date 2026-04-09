from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr, pearsonr
import scikit_posthocs as sp

from ..config import PathConfig
from .metrics import calculate_moe, calculate_mcen


@dataclass
class FriedmanResult:
    chi2: float
    p_value: float
    kendall_w: float
    n_samples: int
    n_models: int
    significant: bool

    def summary(self) -> str:
        lines = [
            "FRIEDMAN TEST",
            "=" * 40,
            f"Chi-squared: {self.chi2:.4f}",
            f"p-value: {self.p_value:.2e}",
            f"Kendall's W (effect size): {self.kendall_w:.4f}",
            f"Samples: {self.n_samples}, Models: {self.n_models}",
            "",
            f"Conclusion: {'Significant (p < 0.05)' if self.significant else 'Not significant'}",
        ]
        return "\n".join(lines)


@dataclass
class CorrelationResult:
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    n_samples: int

    @property
    def is_significant(self) -> bool:
        return self.spearman_p < 0.05

    @property
    def effect_size(self) -> str:
        r = abs(self.spearman_r)
        if r < 0.1:
            return "Negligible"
        elif r < 0.3:
            return "Small"
        elif r < 0.5:
            return "Medium"
        else:
            return "Large"

    def summary(self) -> str:
        lines = [
            "CORRELATION ANALYSIS",
            "=" * 40,
            f"Spearman r: {self.spearman_r:.4f} (p = {self.spearman_p:.4f})",
            f"Pearson r: {self.pearson_r:.4f} (p = {self.pearson_p:.4f})",
            f"Effect size: {self.effect_size}",
            f"Samples: {self.n_samples}",
            "",
            f"Significant: {'Yes' if self.is_significant else 'No'}",
        ]
        return "\n".join(lines)


class StatisticalAnalyzer:
    def __init__(self, path_config: PathConfig):
        self.path_config = path_config

    def _prepare_pivot_table(
        self,
        results_df: pd.DataFrame,
        incomplete_model_handling: str = "drop_incomplete_models",
        min_completion_ratio: float = 0.5,
    ) -> Tuple[pd.DataFrame, List[str]]:
        results_df = results_df.copy()
        results_df["run_id"] = (
            results_df["seed"].astype(str) + "_" + results_df["fold"].astype(str)
        )

        pivot_df = results_df.pivot_table(
            index="run_id",
            columns="model",
            values="f1_score",
            aggfunc="first",
        )

        total_runs = len(pivot_df)
        excluded_models = []

        if incomplete_model_handling == "drop_incomplete_models":
            complete_models = []
            for col in pivot_df.columns:
                completion_ratio = pivot_df[col].notna().sum() / total_runs
                if completion_ratio >= min_completion_ratio:
                    if completion_ratio < 1.0:
                        included_str = 'included' if completion_ratio >= min_completion_ratio else 'excluded'
                        print(
                            f"  WARNING: {col} has {pivot_df[col].notna().sum()}/{total_runs} "
                            f"runs ({completion_ratio * 100:.1f}% complete) - {included_str}"
                        )
                    complete_models.append(col)
                else:
                    excluded_models.append(col)
                    print(
                        f"  EXCLUDING {col}: only {pivot_df[col].notna().sum()}/{total_runs} "
                        f"runs ({completion_ratio * 100:.1f}% complete)"
                    )

            if complete_models:
                pivot_df = pivot_df[complete_models]

            pivot_clean = pivot_df.dropna()

        elif incomplete_model_handling == "drop_runs":
            pivot_clean = pivot_df.dropna()

        elif incomplete_model_handling == "use_available":
            # Keep rows with at least 50% non-NaN values (partial data preserved)
            min_non_nan = max(1, len(pivot_df.columns) // 2)
            pivot_clean = pivot_df.dropna(thresh=min_non_nan)
        else:
            pivot_clean = pivot_df.dropna()

        return pivot_clean, excluded_models

    def friedman_test(
        self,
        results_df: pd.DataFrame,
        incomplete_model_handling: str = "drop_incomplete_models",
    ) -> Tuple[FriedmanResult, pd.DataFrame, List[str]]:
        pivot_clean, excluded_models = self._prepare_pivot_table(
            results_df, incomplete_model_handling
        )

        models = pivot_clean.columns.tolist()
        data = [pivot_clean[m].values for m in models]

        stat, p_value = friedmanchisquare(*data)

        n = len(pivot_clean)
        k = len(models)
        kendall_w = stat / (n * (k - 1))

        result = FriedmanResult(
            chi2=stat,
            p_value=p_value,
            kendall_w=kendall_w,
            n_samples=n,
            n_models=k,
            significant=p_value < 0.05,
        )

        return result, pivot_clean, excluded_models

    def nemenyi_posthoc(self, pivot_df: pd.DataFrame) -> pd.DataFrame:
        # Check rank consistency
        ranks = pivot_df.rank(axis=1, ascending=False)
        rank_std = ranks.std()

        if (rank_std == 0).all():
            print(
                "\n  WARNING: Model rankings are perfectly consistent across all runs!"
            )
            # Show the consistent ranking
            avg_ranks = ranks.mean().sort_values()
            print("\n  Consistent ranking order:")
            for i, (model, rank) in enumerate(avg_ranks.items(), 1):
                print(f"    {i}. {model} (rank {rank:.1f})")
        elif rank_std.max() < 0.5:
            print(
                f"\n  NOTE: Low rank variance detected (max std: {rank_std.max():.3f})"
            )
            print("  Rankings are highly consistent across runs.")

        nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_df.values)
        nemenyi_results.index = pivot_df.columns
        nemenyi_results.columns = pivot_df.columns
        return nemenyi_results

    def calculate_average_ranks(self, pivot_df: pd.DataFrame) -> pd.Series:
        """Calculate average ranks for each model across all runs.

        Args:
            pivot_df: Pivot table with models as columns and runs as rows

        Returns:
            Series with model names as index and average ranks as values
        """
        ranks = pivot_df.rank(axis=1, ascending=False)
        return ranks.mean().sort_values()

    def wilcoxon_pairwise(
        self,
        results_df: pd.DataFrame,
        expected_samples: Optional[int] = None,
        incomplete_model_handling: str = "drop_incomplete_models",
    ) -> Tuple[pd.DataFrame, List[str]]:
        pivot_df, excluded_models = self._prepare_pivot_table(
            results_df, incomplete_model_handling
        )

        models = pivot_df.columns.tolist()
        n_samples = len(pivot_df)

        if expected_samples and n_samples < expected_samples:
            print(
                f"NOTE: Wilcoxon test using {n_samples} samples "
                f"(expected {expected_samples} if all models complete)."
            )

        results = []

        for i, model1 in enumerate(models):
            for model2 in models[i + 1 :]:
                x = pivot_df[model1].values
                y = pivot_df[model2].values

                try:
                    stat, p_value = wilcoxon(x, y, alternative="two-sided")

                    n = len(x)

                    max_rank_sum = n * (n + 1) / 2
                    r = 1 - (2 * stat) / max_rank_sum

                    pooled_std = np.sqrt(
                        (np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2
                    )
                    cohens_d = (
                        (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0
                    )

                    results.append(
                        {
                            "Model 1": model1,
                            "Model 2": model2,
                            "n_samples": n,
                            "Statistic": stat,
                            "p-value": p_value,
                            "Effect Size (r)": r,
                            "Cohen's d": cohens_d,
                            "Significant": p_value < 0.05,
                            "Mean Diff": np.mean(x) - np.mean(y),
                        }
                    )
                except Exception as e:
                    print(f"Error comparing {model1} vs {model2}: {e}")

        return pd.DataFrame(results).sort_values("p-value"), excluded_models

    def correlation_analysis(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> CorrelationResult:
        spearman_r, spearman_p = spearmanr(x, y)
        pearson_r, pearson_p = pearsonr(x, y)

        return CorrelationResult(
            spearman_r=spearman_r,
            spearman_p=spearman_p,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            n_samples=len(x),
        )

    def compute_model_summary(
        self,
        results_df: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        if results_df.empty:
            return pd.DataFrame()

        type_col = "type" if "type" in results_df.columns else "model_type"

        grouped = results_df.groupby(["model", type_col])

        summary_rows = []
        for (model, model_type), group in grouped:
            n = len(group)

            f1_mean = group["f1_score"].mean()
            f1_std = group["f1_score"].std()
            acc_mean = group["accuracy"].mean()
            acc_std = group["accuracy"].std()
            prec_mean = group["precision"].mean()
            prec_std = group["precision"].std()
            rec_mean = group["recall"].mean()
            rec_std = group["recall"].std()
            train_time_mean = group["train_time"].mean()
            inf_time_mean = group["inference_time_ms"].mean()

            moe_f1, z_score = calculate_moe(f1_std, n, confidence_level)
            moe_acc, _ = calculate_moe(acc_std, n, confidence_level)
            moe_prec, _ = calculate_moe(prec_std, n, confidence_level)
            moe_rec, _ = calculate_moe(rec_std, n, confidence_level)

            row = {
                "model": model,
                "type": model_type,
                "n_runs": n,
                "f1_score_mean": round(f1_mean, 4),
                "f1_score_std": round(f1_std, 4),
                "f1_score_moe": round(moe_f1, 4),
                "accuracy_mean": round(acc_mean, 4),
                "accuracy_std": round(acc_std, 4),
                "accuracy_moe": round(moe_acc, 4),
                "precision_mean": round(prec_mean, 4),
                "precision_std": round(prec_std, 4),
                "precision_moe": round(moe_prec, 4),
                "recall_mean": round(rec_mean, 4),
                "recall_std": round(rec_std, 4),
                "recall_moe": round(moe_rec, 4),
                "train_time_mean": round(train_time_mean, 2),
                "inference_time_ms_mean": round(inf_time_mean, 4),
                "confidence_level": confidence_level,
                "z_score": round(z_score, 4),
            }

            has_cm_cols = all(col in group.columns for col in ["tp", "tn", "fp", "fn"])
            if has_cm_cols:
                tp_sum, tn_sum, fp_sum, fn_sum = 0, 0, 0, 0
                mcen_values = []

                for _, result in group.iterrows():
                    if pd.notna(result.get("tp")):
                        tp = int(result["tp"])
                        tn = int(result["tn"])
                        fp = int(result["fp"])
                        fn = int(result["fn"])

                        tp_sum += tp
                        tn_sum += tn
                        fp_sum += fp
                        fn_sum += fn

                        mcen = calculate_mcen(tp, tn, fp, fn)
                        mcen_values.append(mcen)

                if mcen_values:
                    row["mcen_mean"] = round(np.mean(mcen_values), 4)
                    row["mcen_std"] = round(np.std(mcen_values), 4)

                    n_runs = len(mcen_values)
                    row["mcen_aggregated"] = round(
                        calculate_mcen(
                            round(tp_sum / n_runs),
                            round(tn_sum / n_runs),
                            round(fp_sum / n_runs),
                            round(fn_sum / n_runs),
                        ),
                        4,
                    )

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        return summary_df.sort_values("f1_score_mean", ascending=False)

    def run_full_analysis(
        self,
        results_df: pd.DataFrame,
        save_results: bool = True,
        n_seeds: Optional[int] = None,
        n_folds: Optional[int] = None,
        confidence_level: float = 0.95,
        incomplete_model_handling: str = "drop_incomplete_models",
        verbose: bool = False,
    ) -> Dict:
        header = "STATISTICAL ANALYSIS (VERBOSE)" if verbose else "STATISTICAL ANALYSIS"
        print("\n" + "=" * 60)
        print(header)
        print("=" * 60)

        results = {}
        expected_samples = n_seeds * n_folds if n_seeds and n_folds else None

        print(f"\nIncomplete model handling: {incomplete_model_handling}")

        pivot_clean, excluded_models = self._prepare_pivot_table(
            results_df, incomplete_model_handling
        )

        if verbose:
            results_df_copy = results_df.copy()
            results_df_copy["run_id"] = (
                results_df_copy["seed"].astype(str)
                + "_"
                + results_df_copy["fold"].astype(str)
            )
            pivot_df_raw = results_df_copy.pivot_table(
                index="run_id",
                columns="model",
                values="f1_score",
                aggfunc="first",
            )

            print("\n" + "-" * 40)
            print("DATA PASSED TO STATISTICAL TESTS")
            print("-" * 40)
            print(f"\nTotal runs in raw data: {len(pivot_df_raw)}")
            print(f"Runs used in tests: {len(pivot_clean)}")
            if expected_samples:
                print(f"Expected runs (n_seeds x n_folds): {expected_samples}")
            print(f"Total models in raw data: {len(pivot_df_raw.columns)}")
            print(f"Models used in tests: {len(pivot_clean.columns)}")
            print(f"Models included: {list(pivot_clean.columns)}")

            incomplete_models_info = []
            for col in pivot_df_raw.columns:
                missing = pivot_df_raw[col].isna().sum()
                total = len(pivot_df_raw)
                if missing > 0:
                    pct = (total - missing) / total * 100
                    incomplete_models_info.append(
                        f"{col}: {total - missing}/{total} ({pct:.1f}%)"
                    )
            if incomplete_models_info:
                print(
                    f"\nModels with incomplete data: {', '.join(incomplete_models_info)}"
                )

            if excluded_models:
                print(f"EXCLUDED from analysis: {excluded_models}")

            print("\nF1 Scores per model (pivot table used in tests):")
            print(pivot_clean.to_string())

            print("\nDescriptive statistics per model:")
            print(pivot_clean.describe().round(4).to_string())

        print("\n1. Friedman Test")
        print("-" * 40)
        friedman_result, pivot_df, excluded_models = self.friedman_test(
            results_df, incomplete_model_handling
        )
        if excluded_models:
            print(f"Models excluded from statistical tests: {excluded_models}")
        if verbose:
            print(
                f"\nData passed: {len(pivot_df)} observations x {len(pivot_df.columns)} models"
            )
        print(friedman_result.summary())
        results["friedman"] = friedman_result
        results["excluded_models"] = excluded_models

        print("\n2. Nemenyi Post-hoc Test")
        print("-" * 40)
        nemenyi_df = self.nemenyi_posthoc(pivot_df)
        print("\nPairwise p-values:")
        if verbose:
            print(nemenyi_df.round(4).to_string())
        else:
            print(nemenyi_df.round(4))
        results["nemenyi"] = nemenyi_df

        print("\n3. Average Ranks")
        print("-" * 40)
        avg_ranks = self.calculate_average_ranks(pivot_df)
        print("Average Ranks (lower = better):")
        for model, rank in avg_ranks.items():
            print(f"  {model}: {rank:.3f}")
        results["avg_ranks"] = avg_ranks

        print("\n4. Wilcoxon Signed-Rank Tests")
        print("-" * 40)
        wilcoxon_df, _ = self.wilcoxon_pairwise(
            results_df, expected_samples, incomplete_model_handling
        )
        sig_comparisons = wilcoxon_df[wilcoxon_df["Significant"]]
        if verbose:
            print("\nAll pairwise comparisons:")
            print(wilcoxon_df.to_string())
        print(f"\nSignificant pairwise differences: {len(sig_comparisons)}")
        if not wilcoxon_df.empty:
            print(f"Sample size per comparison: {wilcoxon_df['n_samples'].iloc[0]}")
        if verbose and len(sig_comparisons) > 0:
            print("\nSignificant comparisons (p < 0.05):")
            print(sig_comparisons.to_string())
        results["wilcoxon"] = wilcoxon_df

        print("\n5. Model Summary")
        print("-" * 40)
        summary_df = self.compute_model_summary(
            results_df,
            confidence_level=confidence_level,
        )
        print(f"Confidence level: {confidence_level * 100:.0f}%")
        if not summary_df.empty:
            z_score = summary_df["z_score"].iloc[0]
            print(f"Z-score used: {z_score:.4f}")
            print("\nModel Performance Summary:")
            display_cols = ["model", "f1_score_mean", "f1_score_moe"]
            if "mcen_mean" in summary_df.columns:
                display_cols.append("mcen_mean")
            print(summary_df[display_cols].to_string(index=False))
        results["summary"] = summary_df

        if save_results:
            metrics_dir = self.path_config.metrics_dir

            self._save_input_data(results_df, metrics_dir, incomplete_model_handling)

            friedman_dict = {
                "chi2": friedman_result.chi2,
                "p_value": friedman_result.p_value,
                "kendall_w": friedman_result.kendall_w,
                "significant": friedman_result.significant,
                "n_samples": friedman_result.n_samples,
                "n_models": friedman_result.n_models,
            }
            pd.DataFrame([friedman_dict]).to_csv(
                metrics_dir / "friedman_test.csv", index=False
            )

            nemenyi_df.to_csv(metrics_dir / "nemenyi_test.csv")

            wilcoxon_df.to_csv(metrics_dir / "wilcoxon_tests.csv", index=False)

            avg_ranks.to_csv(metrics_dir / "average_ranks.csv", header=["rank"])

            summary_df.to_csv(metrics_dir / "model_summary.csv", index=False)

            print(f"\nResults saved to {metrics_dir}")

        return results

    def _save_input_data(
        self,
        results_df: pd.DataFrame,
        metrics_dir: Path,
        incomplete_model_handling: str = "drop_incomplete_models",
    ) -> None:
        results_copy = results_df.copy()
        results_copy["run_id"] = (
            results_copy["seed"].astype(str) + "_" + results_copy["fold"].astype(str)
        )

        pivot_df = results_copy.pivot_table(
            index="run_id",
            columns="model",
            values="f1_score",
            aggfunc="first",
        )

        pivot_df.to_csv(metrics_dir / "wilcoxon_input_data_raw.csv")

        pivot_clean, excluded = self._prepare_pivot_table(
            results_df, incomplete_model_handling
        )
        pivot_clean.to_csv(metrics_dir / "wilcoxon_input_data.csv")

        print(f"  Input data: {len(pivot_df)} runs, {len(pivot_df.columns)} models")
        print(
            f"  After processing: {len(pivot_clean)} runs, {len(pivot_clean.columns)} models (used in tests)"
        )
        if excluded:
            print(f"  Excluded models: {excluded}")
