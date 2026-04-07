from typing import Any, Dict, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import PathConfig, CLASSICAL_PARAMS, TRANSFORMER_PATHS


def get_model_parameter_count(
    model_name: str, n_features: int = 10000
) -> Dict[str, Any]:
    if model_name in CLASSICAL_PARAMS:
        params = CLASSICAL_PARAMS[model_name]

        if model_name == "LogisticRegression":
            n_params = 2 * (n_features + 1)
            details = f"2 classes × ({n_features} weights + 1 bias)"

        elif model_name == "NaiveBayes":
            n_params = 2 * n_features + 2
            details = f"2 × {n_features} feature probs + 2 class priors"

        elif model_name == "SVM":
            n_params = 2 * (n_features + 1)
            details = f"2 classes × ({n_features} weights + 1 bias)"

        elif model_name == "XGBoost":
            n_estimators = params.get("n_estimators", 100)
            max_depth = params.get("max_depth", 6)
            max_leaves = 2**max_depth
            internal_nodes = max_leaves - 1
            params_per_tree = internal_nodes * 2 + max_leaves
            n_params = n_estimators * params_per_tree
            details = f"{n_estimators} trees × {params_per_tree} params/tree"

        elif model_name == "LightGBM":
            n_estimators = params.get("n_estimators", 100)
            num_leaves = params.get("num_leaves", 31)
            internal_nodes = num_leaves - 1
            params_per_tree = internal_nodes * 2 + num_leaves
            n_params = n_estimators * params_per_tree
            details = f"{n_estimators} trees × {params_per_tree} params/tree"

        else:
            n_params = 0
            details = "Unknown"

        return {
            "n_params": n_params,
            "details": details,
            "type": "Classical",
        }

    elif model_name in TRANSFORMER_PATHS:
        if model_name == "ELECTRA":
            n_params = 13_548_800  # ~13.5M parameters
            details = "ELECTRA-small discriminator"
        elif model_name == "RoBERTa":
            n_params = 124_645_632  # ~125M parameters
            details = "RoBERTa-base"
        else:
            n_params = 0
            details = "Unknown transformer"

        return {
            "n_params": n_params,
            "details": details,
            "type": "Transformer",
        }

    return {"n_params": 0, "details": "Unknown model", "type": "Unknown"}


def format_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    else:
        return str(n)


class Visualizer:
    def __init__(self, path_config: PathConfig):
        self.path_config = path_config
        self.figures_dir = path_config.figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"

    def analyze_data_distribution(self, df: pd.DataFrame) -> None:
        if "app_name" not in df.columns and "game_name" in df.columns:
            df = df.copy()
            df["app_name"] = df["game_name"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax1 = axes[0]
        class_counts = df["positive"].value_counts().sort_index(ascending=False)
        labels = ["Positive", "Negative"]
        colors = ["#2ecc71", "#e74c3c"]
        total = class_counts.sum()
        bars = ax1.bar(labels, class_counts.values, color=colors, edgecolor="black")
        ax1.set_ylabel("Number of Reviews", fontsize=11)
        for bar, count in zip(bars, class_counts.values):
            pct = count / total * 100
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        ax2 = axes[1]
        game_col = "app_name" if "app_name" in df.columns else "game_name"
        game_counts = df[game_col].value_counts()
        game_counts.plot(kind="barh", ax=ax2, color="steelblue", edgecolor="black")
        ax2.set_xlabel("Number of Reviews", fontsize=11)
        ax2.invert_yaxis()

        ax3 = axes[2]
        text_col = "cleaned_text" if "cleaned_text" in df.columns else "review_text"
        df_plot = df.copy()
        df_plot["text_length"] = df_plot[text_col].astype(str).str.len()
        df_plot["text_length"].hist(
            bins=50, ax=ax3, color="purple", alpha=0.7, edgecolor="black"
        )
        ax3.set_xlabel("Review Length (characters)", fontsize=11)
        ax3.set_ylabel("Frequency", fontsize=11)
        median_len = df_plot["text_length"].median()
        ax3.axvline(
            median_len, color="red", linestyle="--", label=f"Median: {median_len:.0f}"
        )
        ax3.legend()

        plt.tight_layout()
        save_path = self.figures_dir / "data_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

        print(f"\n{'=' * 60}")
        print("DETAILED DATA STATISTICS")
        print(f"{'=' * 60}")
        print(f"\nReview text length:")
        print(f"  Mean: {df_plot['text_length'].mean():.1f} characters")
        print(f"  Median: {df_plot['text_length'].median():.1f} characters")
        print(f"  Min: {df_plot['text_length'].min()} characters")
        print(f"  Max: {df_plot['text_length'].max()} characters")

        if "review_type" in df.columns:
            print(f"\nReview type distribution:")
            for review_type in df["review_type"].unique():
                subset = df[df["review_type"] == review_type]
                pos_pct = subset["positive"].mean() * 100
                print(
                    f"  {review_type}: {len(subset):,} reviews ({
                        pos_pct:.1f}% positive)"
                )

    def plot_early_access_vs_post_release(self, df: pd.DataFrame) -> None:
        """Compare early access and post-release sentiment distributions."""
        if "review_type" not in df.columns:
            print("No review_type column found - skipping EA vs PR comparison")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        review_type_sentiment = (
            df.groupby(["review_type", "positive"]).size().unstack(fill_value=0)
        )
        review_type_sentiment_pct = (
            review_type_sentiment.div(review_type_sentiment.sum(axis=1), axis=0) * 100
        )

        ax1 = axes[0]
        review_type_sentiment_pct.plot(
            kind="bar", ax=ax1, color=["#e74c3c", "#27ae60"], edgecolor="white"
        )
        ax1.set_xlabel("Review Type", fontsize=11)
        ax1.set_ylabel("Percentage", fontsize=11)
        ax1.legend(["Negative", "Positive"])
        ax1.tick_params(axis="x", rotation=0)

        ax2 = axes[1]
        review_type_counts = df["review_type"].value_counts()
        bars = ax2.bar(
            review_type_counts.index,
            review_type_counts.values,
            color=["#3498db", "#9b59b6"],
        )
        ax2.set_xlabel("Review Type", fontsize=11)
        ax2.set_ylabel("Number of Reviews", fontsize=11)
        for bar, count in zip(bars, review_type_counts.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        save_path = self.figures_dir / "early_access_vs_post_release.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_model_comparison(
        self,
        summary_df: pd.DataFrame,
        n_seeds: int = 12,
        n_folds: int = 5,
        incomplete_models: Optional[Dict[str, str]] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 7))

        VISUAL_ERROR_SCALE = 20

        if "f1_score_mean" in summary_df.columns:
            f1_col = "f1_score_mean"
            err_col = (
                "f1_score_moe"
                if "f1_score_moe" in summary_df.columns
                else "f1_score_std"
            )
            std_col = "f1_score_std"
        elif "f1_mean" in summary_df.columns:
            f1_col = "f1_mean"
            err_col = "f1_std"
            std_col = "f1_std"
        elif "f1_score" in summary_df.columns:
            summary_df = (
                summary_df.groupby(["model", "type"])
                .agg({"f1_score": ["mean", "std"]})
                .reset_index()
            )
            summary_df.columns = ["model", "type", "f1_mean", "f1_std"]
            f1_col = "f1_mean"
            err_col = "f1_std"
            std_col = "f1_std"
        else:
            print("Could not find F1 score column")
            return

        summary_sorted = summary_df.sort_values(f1_col, ascending=True)

        type_col = "model_type" if "model_type" in summary_sorted.columns else "type"

        colors = [
            "#e74c3c" if t == "Transformer" else "#3498db"
            for t in summary_sorted[type_col]
        ]

        y_pos = np.arange(len(summary_sorted))

        bars = ax.barh(
            y_pos,
            summary_sorted[f1_col],
            color=colors,
            alpha=0.9,
            height=0.3,
        )

        err_values = summary_sorted.get(err_col, None)

        if err_values is not None:
            err_array = (
                err_values.values if hasattr(err_values, "values") else err_values
            )
            for i, (y, x_center, actual_err) in enumerate(
                zip(y_pos, summary_sorted[f1_col].values, err_array)
            ):
                if actual_err > 0:
                    vis_err = actual_err * VISUAL_ERROR_SCALE

                    ax.plot(
                        [x_center - vis_err, x_center + vis_err],
                        [y, y],
                        color="black",
                        linewidth=1.5,
                        solid_capstyle="butt",
                        zorder=5,
                    )
                    cap_height = 0.15
                    ax.plot(
                        [x_center - vis_err, x_center - vis_err],
                        [y - cap_height, y + cap_height],
                        color="black",
                        linewidth=1,
                        zorder=5,
                    )
                    ax.plot(
                        [x_center + vis_err, x_center + vis_err],
                        [y - cap_height, y + cap_height],
                        color="black",
                        linewidth=1,
                        zorder=5,
                    )
        y_labels = []
        for model in summary_sorted["model"]:
            if incomplete_models and model in incomplete_models:
                y_labels.append(f"{model}*")
            else:
                y_labels.append(model)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=11)
        ax.set_xlabel("F1-Score (Macro)", fontsize=12, fontweight="bold")

        err_series = summary_sorted.get(err_col, pd.Series([0] * len(summary_sorted)))
        std_series = summary_sorted.get(std_col, pd.Series([0] * len(summary_sorted)))

        for i, (val, err, std) in enumerate(
            zip(summary_sorted[f1_col], err_series, std_series)
        ):
            if "moe" in err_col.lower() and err > 0:
                label = f"{val:.4f} ± {err:.4f}"
            elif std > 0:
                label = f"{val:.4f} ± {std:.4f}"
            else:
                label = f"{val:.4f}"

            vis_err_offset = (err * VISUAL_ERROR_SCALE) if err > 0 else 0

            ax.text(
                val + vis_err_offset + 0.005,
                i,
                label,
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        trans_patch = mpatches.Patch(color="#e74c3c", label="Transformer")
        class_patch = mpatches.Patch(color="#3498db", label="Classical ML")

        from matplotlib.lines import Line2D

        err_line = Line2D(
            [0],
            [0],
            color="black",
            lw=2,
            label=f"Error Bars (Magnified {VISUAL_ERROR_SCALE}x)",
        )

        ax.legend(
            handles=[trans_patch, class_patch, err_line],
            loc="lower right",
            fontsize=10,
            frameon=True,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        min_score = summary_sorted[f1_col].min()
        if min_score > 0.8:
            ax.set_xlim(min_score - 0.05, 1.02)  # Add some buffer

        if incomplete_models:
            note_lines = [f"* {m}: {n}" for m, n in incomplete_models.items()]
            note_text = "\n".join(note_lines)
            ax.text(
                0.02,
                0.02,
                note_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                style="italic",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        save_path = self.figures_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_boxplots(
        self,
        results_df: pd.DataFrame,
        incomplete_models: Optional[Dict[str, str]] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 7))

        f1_col = "f1_score" if "f1_score" in results_df.columns else "f1"
        type_col = "type" if "type" in results_df.columns else "model_type"

        model_order = (
            results_df.groupby("model")[f1_col]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

        model_types = (
            results_df.drop_duplicates("model").set_index("model")[type_col].to_dict()
        )

        palette = {
            m: "#e74c3c" if model_types.get(m) == "Transformer" else "#3498db"
            for m in model_order
        }

        sns.boxplot(
            data=results_df,
            x="model",
            y=f1_col,
            order=model_order,
            palette=palette,
            ax=ax,
        )

        sns.stripplot(
            data=results_df,
            x="model",
            y=f1_col,
            order=model_order,
            color="black",
            alpha=0.3,
            size=3,
            ax=ax,
        )

        x_labels = []
        for m in model_order:
            if incomplete_models and m in incomplete_models:
                x_labels.append(f"{m}*")
            else:
                x_labels.append(m)
        ax.set_xticklabels(x_labels)

        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("F1-Score (Macro)", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

        trans_patch = mpatches.Patch(color="#e74c3c", label="Transformer")
        class_patch = mpatches.Patch(color="#3498db", label="Classical ML")
        ax.legend(handles=[trans_patch, class_patch], loc="upper right", fontsize=10)

        if incomplete_models:
            note_lines = [f"* {m}: {n}" for m, n in incomplete_models.items()]
            note_text = "\n".join(note_lines)
            ax.text(
                0.02,
                0.02,
                note_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                style="italic",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        save_path = self.figures_dir / "boxplots.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_nemenyi_heatmap(self, nemenyi_df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(11, 9))

        annot_matrix = nemenyi_df.copy()
        annot_labels = np.empty_like(annot_matrix, dtype=object)

        for i in range(len(nemenyi_df)):
            for j in range(len(nemenyi_df.columns)):
                p_val = nemenyi_df.iloc[i, j]
                if i == j:
                    annot_labels[i, j] = "-"
                elif p_val < 0.001:
                    annot_labels[i, j] = f"{p_val:.2e}\n***"
                elif p_val < 0.01:
                    annot_labels[i, j] = f"{p_val:.4f}\n**"
                elif p_val < 0.05:
                    annot_labels[i, j] = f"{p_val:.4f}\n*"
                else:
                    annot_labels[i, j] = f"{p_val:.4f}\nns"

        mask = np.triu(np.ones_like(nemenyi_df, dtype=bool), k=1)

        sns.heatmap(
            nemenyi_df,
            annot=annot_labels,
            fmt="",
            cmap="RdYlGn_r",  # Reverse so red=significant, green=not significant
            center=0.05,
            vmin=0,
            vmax=0.1,  # Focus on region around alpha=0.05
            square=True,
            linewidths=1,
            cbar_kws={"label": "p-value", "extend": "max"},
            mask=mask,
            ax=ax,
        )

        ax.set_xlabel("Model", fontsize=11, fontweight="bold")
        ax.set_ylabel("Model", fontsize=11, fontweight="bold")

        plt.tight_layout()
        save_path = self.figures_dir / "nemenyi_heatmap.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_thesis_validation(
        self,
        game_sentiment: pd.DataFrame,
        correlation_result: Optional["CorrelationResult"] = None,
    ) -> None:
        """Create scatter plot for thesis validation (EA sentiment vs post-release success)."""
        valid_games = game_sentiment.dropna(subset=["pr_actual_positive_ratio"])

        if len(valid_games) < 3:
            print(
                "Not enough games with both EA and post-release data for visualization"
            )
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        ea_pred = valid_games["ea_predicted_positive_ratio"].values
        pr_actual = valid_games["pr_actual_positive_ratio"].values

        ax.scatter(
            ea_pred, pr_actual, s=100, alpha=0.7, c="steelblue", edgecolors="black"
        )

        for _, row in valid_games.iterrows():
            ax.annotate(
                row["game_name"][:15],
                (row["ea_predicted_positive_ratio"], row["pr_actual_positive_ratio"]),
                fontsize=8,
                alpha=0.7,
            )

        z = np.polyfit(ea_pred, pr_actual, 1)
        p = np.poly1d(z)

        if correlation_result:
            corr_label = f"r={correlation_result.spearman_r:.2f}"
        else:
            from scipy.stats import spearmanr

            r, _ = spearmanr(ea_pred, pr_actual)
            corr_label = f"r={r:.2f}"

        ax.plot(ea_pred, p(ea_pred), "r--", alpha=0.8, label=f"Trend ({corr_label})")

        ax.set_xlabel("Early Access Predicted Positive Ratio", fontsize=12)
        ax.set_ylabel("Post-Release Actual Positive Ratio", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.figures_dir / "thesis_validation.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_wilcoxon_effects(self, wilcoxon_df: pd.DataFrame) -> None:
        """Create dual visualization of Wilcoxon test results with both r and Cohen's d."""
        if len(wilcoxon_df) == 0:
            print("No Wilcoxon results to plot")
            return

        has_cohens_d = "Cohen's d" in wilcoxon_df.columns

        if has_cohens_d:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        else:
            fig, ax1 = plt.subplots(figsize=(14, 8))

        wilcoxon_df = wilcoxon_df.copy()
        wilcoxon_df["Comparison"] = (
            wilcoxon_df["Model 1"] + "\nvs\n" + wilcoxon_df["Model 2"]
        )

        wilcoxon_df["AbsMeanDiff"] = wilcoxon_df["Mean Diff"].abs()
        sorted_df = wilcoxon_df.sort_values("AbsMeanDiff", ascending=True)

        colors = ["#27ae60" if sig else "#bdc3c7" for sig in sorted_df["Significant"]]

        y_pos = range(len(sorted_df))
        ax1.barh(
            y_pos,
            sorted_df["Effect Size (r)"],
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )

        ax1.axvline(
            x=0.5, color="#e74c3c", linestyle="--", label="Large (0.5)", alpha=0.7
        )
        ax1.axvline(
            x=1.0,
            color="#c0392b",
            linestyle="-",
            linewidth=2,
            label="Complete Domination (1.0)",
            alpha=0.7,
        )

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_df["Comparison"], fontsize=7)
        ax1.set_xlabel(
            "Effect Size r (Rank-Biserial Correlation)", fontsize=11, fontweight="bold"
        )
        ax1.legend(loc="lower right", fontsize=9)
        ax1.set_xlim(0, min(1.1, sorted_df["Effect Size (r)"].max() * 1.05))
        ax1.grid(axis="x", alpha=0.3)

        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            r_val = row["Effect Size (r)"]
            label = f"{r_val:.2f}"
            ax1.text(r_val + 0.02, i, label, va="center", fontsize=8, fontweight="bold")

        if has_cohens_d:
            cohens_d_vals = sorted_df["Cohen's d"].abs()
            colors_d = []
            for val in cohens_d_vals:
                if val >= 0.8:
                    colors_d.append("#c0392b")  # Very large
                elif val >= 0.5:
                    colors_d.append("#e74c3c")  # Large
                elif val >= 0.2:
                    colors_d.append("#f39c12")  # Small
                else:
                    colors_d.append("#95a5a6")  # Negligible

            ax2.barh(y_pos, cohens_d_vals, color=colors_d, alpha=0.8, edgecolor="black")

            ax2.axvline(
                x=0.2, color="#f39c12", linestyle="--", label="Small (0.2)", alpha=0.7
            )
            ax2.axvline(
                x=0.5, color="#e67e22", linestyle="--", label="Medium (0.5)", alpha=0.7
            )
            ax2.axvline(
                x=0.8, color="#e74c3c", linestyle="--", label="Large (0.8)", alpha=0.7
            )

            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(sorted_df["Comparison"], fontsize=7)
            ax2.set_xlabel(
                "Cohen's d (Standardized Effect Size)", fontsize=11, fontweight="bold"
            )
            ax2.legend(loc="lower right", fontsize=9)
            ax2.set_xlim(0, cohens_d_vals.max() * 1.1)
            ax2.grid(axis="x", alpha=0.3)

            for i, val in enumerate(cohens_d_vals):
                label = f"{val:.1f}"
                ax2.text(
                    val + cohens_d_vals.max() * 0.02,
                    i,
                    label,
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

        plt.tight_layout()
        save_path = self.figures_dir / "wilcoxon_effects.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def create_results_table_image(
        self,
        summary_df: pd.DataFrame,
        n_seeds: int = 12,
        n_folds: int = 5,
        n_features: int = 10000,
        incomplete_models: Optional[Dict[str, str]] = None,
    ) -> None:
        """Create a clean results table as an image.

        Includes parameter count, MOE and optionally MCEN if available in summary_df.

        Args:
            summary_df: DataFrame with model summary statistics
            n_seeds: Number of seeds used
            n_folds: Number of folds used
            n_features: Number of TF-IDF features (for classical models)
            incomplete_models: Dict mapping model name to note about incomplete data
        """
        fig = plt.figure(figsize=(20, 9))
        ax = fig.add_subplot(111)
        ax.axis("off")

        if "f1_score_mean" in summary_df.columns:
            f1_col = "f1_score_mean"
            std_col = "f1_score_std"
            moe_col = "f1_score_moe" if "f1_score_moe" in summary_df.columns else None
        elif "f1_mean" in summary_df.columns:
            f1_col = "f1_mean"
            std_col = "f1_std"
            moe_col = None
        else:
            f1_col = "f1_score"
            std_col = None
            moe_col = None

        type_col = "model_type" if "model_type" in summary_df.columns else "type"
        has_mcen = "mcen_mean" in summary_df.columns

        sorted_df = summary_df.sort_values(f1_col, ascending=False)

        table_data = []
        for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
            std_val = row.get(std_col, 0) if std_col else 0
            moe_val = row.get(moe_col, 0) if moe_col else 0

            model_name = row["model"]
            param_info = get_model_parameter_count(model_name, n_features)
            param_str = format_param_count(param_info["n_params"])

            display_name = model_name
            if incomplete_models and model_name in incomplete_models:
                display_name = f"{model_name}*"

            row_data = [
                f"{rank}",
                display_name,
                row.get(type_col, "Unknown"),
                param_str,
                f"{row[f1_col]:.4f}",
            ]

            if moe_val > 0:
                row_data.append(f"±{moe_val:.4f}")
            elif std_val > 0:
                row_data.append(f"±{std_val:.4f}")
            else:
                row_data.append("N/A")

            row_data.extend(
                [
                    f"{row.get('accuracy_mean', row.get('accuracy', 0)):.4f}",
                    f"{row.get('precision_mean', row.get('precision', 0)):.4f}",
                    f"{row.get('recall_mean', row.get('recall', 0)):.4f}",
                ]
            )

            if has_mcen:
                mcen_val = row.get("mcen_mean", 0)
                row_data.append(f"{mcen_val:.4f}" if mcen_val > 0 else "N/A")

            row_data.append(
                f"{row.get('train_time_mean', row.get('train_time', 0)):.2f}s"
            )
            table_data.append(row_data)

        err_label = "MOE" if moe_col else "Std Dev"
        headers = [
            "#",
            "Model",
            "Type",
            "Params",
            "Avg F1",
            err_label,
            "Avg Acc",
            "Avg Prec",
            "Avg Recall",
        ]
        if has_mcen:
            headers.append("MCEN")
        headers.append("Avg Time")

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 0.85],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 3)

        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(weight="bold", color="white", fontsize=11)
            cell.set_edgecolor("white")
            cell.set_linewidth(2)

        for i in range(1, len(table_data) + 1):
            if i <= 2:
                bg_color = "#fff3cd"  # Highlight top 2
                text_weight = "bold"
            else:
                bg_color = "#f8f9fa" if i % 2 == 0 else "white"
                text_weight = "normal"

            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(bg_color)
                cell.set_text_props(weight=text_weight, fontsize=9)
                cell.set_edgecolor("#dee2e6")
                cell.set_linewidth(1)

        conf_level = summary_df.get("confidence_level", pd.Series([0.95])).iloc[0]
        conf_label = f" ({conf_level * 100:.0f}% CI)" if moe_col else ""

        fig.text(
            0.5,
            0.91,
            f"Average results from {n_seeds * n_folds} runs ({n_seeds} seeds x {
                n_folds
            } folds per model){conf_label}",
            ha="center",
            fontsize=12,
            style="italic",
            color="#555",
        )

        if incomplete_models:
            note_lines = [f"* {m}: {n}" for m, n in incomplete_models.items()]
            note_text = "  |  ".join(note_lines)
            fig.text(
                0.5,
                0.02,
                note_text,
                ha="center",
                fontsize=9,
                style="italic",
                color="#666",
            )

        plt.tight_layout()
        save_path = self.figures_dir / "results_table.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def generate_all_visualizations(
        self,
        data_df: Optional[pd.DataFrame] = None,
        results_df: Optional[pd.DataFrame] = None,
        summary_df: Optional[pd.DataFrame] = None,
        nemenyi_df: Optional[pd.DataFrame] = None,
        wilcoxon_df: Optional[pd.DataFrame] = None,
        game_sentiment: Optional[pd.DataFrame] = None,
        n_seeds: int = 12,
        n_folds: int = 5,
        incomplete_models: Optional[Dict[str, str]] = None,
    ) -> None:
        """Generate all available visualizations.

        Args:
            data_df: Raw data DataFrame for distribution analysis
            results_df: Results DataFrame with per-run metrics
            summary_df: Summary DataFrame with aggregated metrics
            nemenyi_df: Nemenyi post-hoc test p-value matrix
            wilcoxon_df: Wilcoxon pairwise test results
            game_sentiment: Game-level sentiment predictions for thesis validation
            n_seeds: Number of seeds used in training
            n_folds: Number of folds used in cross-validation
            incomplete_models: Dict mapping model name to note about incomplete data
                e.g., {"ELECTRA": "10 real runs, 50 mean-imputed"}
        """
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        if incomplete_models:
            print("\nNote: Some models have incomplete data:")
            for model, note in incomplete_models.items():
                print(f"  - {model}: {note}")

        if data_df is not None:
            print("\n1. Data Distribution Analysis")
            self.analyze_data_distribution(data_df)
            self.plot_early_access_vs_post_release(data_df)

        if summary_df is not None:
            print("\n2. Model Comparison")
            self.plot_model_comparison(summary_df, n_seeds, n_folds, incomplete_models)
            self.create_results_table_image(
                summary_df, n_seeds, n_folds, incomplete_models=incomplete_models
            )

        if results_df is not None:
            print("\n3. Box Plots")
            self.plot_boxplots(results_df, incomplete_models)

        if nemenyi_df is not None:
            print("\n4. Nemenyi Heatmap")
            self.plot_nemenyi_heatmap(nemenyi_df)

        if wilcoxon_df is not None:
            print("\n5. Wilcoxon Effects")
            self.plot_wilcoxon_effects(wilcoxon_df)

        if game_sentiment is not None:
            print("\n6. Thesis Validation")
            self.plot_thesis_validation(game_sentiment)

        print(f"\nAll visualizations saved to: {self.figures_dir}")

    def plot_success_correlations(
        self,
        game_sentiment: pd.DataFrame,
        success_metrics: pd.DataFrame,
        success_correlations: Optional[Dict] = None,
    ) -> None:
        """Create 2x3 grid of scatter plots for success metric correlations.

        Plots:
        - EA Sentiment vs Metacritic
        - EA Sentiment vs Estimated Sales
        - EA Sentiment vs Playtime
        - EA Sentiment vs Review Score
        - EA Sentiment vs Post-Release Sentiment
        """
        if "app_id" in game_sentiment.columns:
            merged = game_sentiment.merge(
                success_metrics, on="app_id", how="inner", suffixes=("", "_metrics")
            )
        else:
            game_sentiment = game_sentiment.copy()
            success_metrics = success_metrics.copy()
            game_sentiment["name_normalized"] = (
                game_sentiment["game_name"].str.lower().str.strip()
            )
            success_metrics["name_normalized"] = (
                success_metrics["app_name"].str.lower().str.strip()
            )
            merged = game_sentiment.merge(
                success_metrics,
                on="name_normalized",
                how="inner",
                suffixes=("", "_metrics"),
            )

        if len(merged) < 3:
            print("Not enough matched games for success correlation plots")
            return

        if "owners_midpoint" not in merged.columns:
            merged["owners_midpoint"] = (
                merged["steamspy_owners_min"] + merged["steamspy_owners_max"]
            ) / 2

        if "review_score" not in merged.columns:
            merged["review_score"] = merged["steamspy_positive"] / (
                merged["steamspy_positive"] + merged["steamspy_negative"]
            )

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        plots = [
            (
                "steam_metacritic_score",
                "EA Sentiment vs Metacritic",
                "Metacritic Score",
                False,
            ),
            (
                "owners_midpoint",
                "EA Sentiment vs Estimated Sales",
                "Estimated Owners",
                True,
            ),
            (
                "steamspy_avg_playtime",
                "EA Sentiment vs Playtime",
                "Avg Playtime (min)",
                True,
            ),
            (
                "review_score",
                "EA Sentiment vs Review Score",
                "Review Positive Ratio",
                False,
            ),
            (
                "pr_actual_positive_ratio",
                "EA Sentiment vs Post-Release Sentiment",
                "Post-Release Positive Ratio",
                False,
            ),
            (None, None, None, None),  # Empty plot slot
        ]

        ea_sentiment = merged["ea_predicted_positive_ratio"]

        for ax, (col, title, ylabel, log_scale) in zip(axes.flatten(), plots):
            if col is None:
                ax.axis("off")
                continue

            if col not in merged.columns:
                ax.text(
                    0.5,
                    0.5,
                    f"{col}\nnot available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            valid = pd.notna(merged[col]) & pd.notna(ea_sentiment)
            if valid.sum() < 3:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            x = ea_sentiment[valid].values
            y = merged[col][valid].values
            names = (
                merged.loc[valid, "game_name"].values
                if "game_name" in merged.columns
                else None
            )

            ax.scatter(x, y, s=80, alpha=0.7, c="steelblue", edgecolors="black")

            if names is not None:
                for xi, yi, name in zip(x, y, names):
                    label = name[:12] + "..." if len(name) > 12 else name
                    ax.annotate(
                        label,
                        (xi, yi),
                        fontsize=7,
                        alpha=0.7,
                        xytext=(3, 3),
                        textcoords="offset points",
                    )

            from scipy.stats import spearmanr

            r, p = spearmanr(x, y)
            z = np.polyfit(x, y, 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(
                x_line, p_line(x_line), "r--", alpha=0.7, label=f"r={r:.2f}, p={p:.3f}"
            )

            ax.set_xlabel("EA Predicted Positive Ratio", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

            if log_scale and y.min() > 0:
                ax.set_yscale("log")

        plt.tight_layout()

        save_path = self.figures_dir / "success_correlations.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_correlation_matrix(
        self,
        game_sentiment: pd.DataFrame,
        success_metrics: pd.DataFrame,
    ) -> None:
        """Create heatmap showing EA predicted sentiment vs all success metrics.

        This focuses on the thesis question: Does MODEL-PREDICTED EA sentiment
        correlate with success metrics? We exclude ea_actual since:
        1. You already know the actual labels - no prediction needed
        2. The thesis is about using a MODEL to predict success
        """
        if "app_id" in game_sentiment.columns:
            merged = game_sentiment.merge(
                success_metrics, on="app_id", how="inner", suffixes=("", "_metrics")
            )
        else:
            game_sentiment = game_sentiment.copy()
            success_metrics = success_metrics.copy()
            game_sentiment["name_normalized"] = (
                game_sentiment["game_name"].str.lower().str.strip()
            )
            success_metrics["name_normalized"] = (
                success_metrics["app_name"].str.lower().str.strip()
            )
            merged = game_sentiment.merge(
                success_metrics,
                on="name_normalized",
                how="inner",
                suffixes=("", "_metrics"),
            )

        if len(merged) < 5:
            print("Not enough matched games for correlation matrix")
            return

        if "owners_midpoint" not in merged.columns:
            merged["owners_midpoint"] = (
                merged["steamspy_owners_min"] + merged["steamspy_owners_max"]
            ) / 2

        if "review_score" not in merged.columns:
            merged["review_score"] = merged["steamspy_positive"] / (
                merged["steamspy_positive"] + merged["steamspy_negative"]
            )

        cols = [
            "ea_predicted_positive_ratio",
            "pr_actual_positive_ratio",
            "steam_metacritic_score",
            "owners_midpoint",
            "steamspy_avg_playtime",
            "review_score",
        ]

        available_cols = [c for c in cols if c in merged.columns]
        corr_data = merged[available_cols].copy()

        rename_map = {
            "ea_predicted_positive_ratio": "EA Sentiment\n(Model Prediction)",
            "pr_actual_positive_ratio": "Post-Release\nSentiment",
            "steam_metacritic_score": "Metacritic\nScore",
            "owners_midpoint": "Estimated\nSales",
            "steamspy_avg_playtime": "Average\nPlaytime",
            "review_score": "Overall\nReview Score",
        }
        corr_data.columns = [rename_map.get(c, c) for c in corr_data.columns]

        from scipy.stats import spearmanr

        n = len(corr_data.columns)
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))

        for i, col1 in enumerate(corr_data.columns):
            for j, col2 in enumerate(corr_data.columns):
                valid = pd.notna(corr_data[col1]) & pd.notna(corr_data[col2])
                if valid.sum() >= 3:
                    r, p = spearmanr(corr_data[col1][valid], corr_data[col2][valid])
                    corr_matrix[i, j] = r
                    p_matrix[i, j] = p
                else:
                    corr_matrix[i, j] = np.nan
                    p_matrix[i, j] = np.nan

        corr_df = pd.DataFrame(
            corr_matrix, index=corr_data.columns, columns=corr_data.columns
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        sns.heatmap(
            corr_df,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Spearman r"},
            mask=mask,
            ax=ax,
        )

        plt.tight_layout()

        save_path = self.figures_dir / "correlation_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_feature_metric_grid(
        self,
        game_sentiment: pd.DataFrame,
        success_metrics: pd.DataFrame,
    ) -> None:
        """Render the full sentiment-feature x success-metric correlation heatmap.

        Each cell shows Spearman r between one EA-derived sentiment feature
        (rows) and one commercial-success metric (columns). Cells are
        annotated with r and a significance marker (* p<0.05, ** p<0.01,
        *** p<0.001). This is the visual analogue of the correlation table
        printed by ``correlate_with_success_metrics``.
        """
        from scipy.stats import spearmanr

        if "app_id" in game_sentiment.columns:
            merged = game_sentiment.merge(
                success_metrics, on="app_id", how="inner", suffixes=("", "_metrics")
            )
        else:
            game_sentiment = game_sentiment.copy()
            success_metrics = success_metrics.copy()
            game_sentiment["name_normalized"] = (
                game_sentiment["game_name"].str.lower().str.strip()
            )
            success_metrics["name_normalized"] = (
                success_metrics["app_name"].str.lower().str.strip()
            )
            merged = game_sentiment.merge(
                success_metrics,
                on="name_normalized",
                how="inner",
                suffixes=("", "_metrics"),
            )

        if len(merged) < 5:
            print("Not enough matched games for feature-metric grid")
            return

        # Derive convenience columns to match correlate_with_success_metrics
        if "owners_midpoint" not in merged.columns and (
            "steamspy_owners_min" in merged.columns
            and "steamspy_owners_max" in merged.columns
        ):
            merged["owners_midpoint"] = (
                merged["steamspy_owners_min"] + merged["steamspy_owners_max"]
            ) / 2

        if "review_score" not in merged.columns and (
            "steamspy_positive" in merged.columns
            and "steamspy_negative" in merged.columns
        ):
            merged["review_score"] = merged["steamspy_positive"] / (
                merged["steamspy_positive"] + merged["steamspy_negative"]
            )

        if "ea_review_count" in merged.columns:
            merged["log_ea_review_count"] = np.log1p(merged["ea_review_count"])

        sentiment_features = [
            ("ea_predicted_positive_ratio", "Predicted Sentiment"),
            ("ea_sentiment_std", "Sentiment Variance"),
            ("ea_mean_review_length", "Mean Review Length"),
            ("ea_long_review_ratio", "Long Review Ratio"),
            ("ea_neg_pos_length_ratio", "Neg/Pos Length Ratio"),
            ("log_ea_review_count", "Log Review Count"),
        ]

        success_metrics_config = [
            ("owners_midpoint", "Estimated Sales\n(Owners)"),
            ("estimated_revenue_usd", "Estimated\nRevenue"),
            ("review_score", "Post-Release\nReview Score"),
            ("steam_metacritic_score", "Metacritic\nScore"),
            ("steamspy_avg_playtime", "Average\nPlaytime"),
            ("pr_actual_positive_ratio", "Post-Release\nSentiment"),
        ]

        feat_rows = [(c, n) for c, n in sentiment_features if c in merged.columns]
        metric_cols = [(c, n) for c, n in success_metrics_config if c in merged.columns]

        if not feat_rows or not metric_cols:
            print("No sentiment features or success metrics available for grid")
            return

        n_rows = len(feat_rows)
        n_cols = len(metric_cols)
        r_grid = np.full((n_rows, n_cols), np.nan)
        p_grid = np.full((n_rows, n_cols), np.nan)
        n_grid = np.zeros((n_rows, n_cols), dtype=int)

        for i, (fcol, _) in enumerate(feat_rows):
            for j, (mcol, _) in enumerate(metric_cols):
                valid = pd.notna(merged[fcol]) & pd.notna(merged[mcol])
                if valid.sum() < 5:
                    continue
                r, p = spearmanr(merged[fcol][valid], merged[mcol][valid])
                r_grid[i, j] = r
                p_grid[i, j] = p
                n_grid[i, j] = valid.sum()

        def _stars(p: float) -> str:
            if np.isnan(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        annot = np.empty_like(r_grid, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                if np.isnan(r_grid[i, j]):
                    annot[i, j] = "n/a"
                else:
                    annot[i, j] = f"{r_grid[i, j]:.2f}{_stars(p_grid[i, j])}"

        corr_df = pd.DataFrame(
            r_grid,
            index=[n for _, n in feat_rows],
            columns=[n for _, n in metric_cols],
        )

        fig, ax = plt.subplots(figsize=(max(10, 1.6 * n_cols), max(6, 0.9 * n_rows)))

        sns.heatmap(
            corr_df,
            annot=annot,
            fmt="",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=False,
            linewidths=0.5,
            cbar_kws={"label": "Spearman r"},
            ax=ax,
        )

        ax.set_title(
            "Sentiment Features vs Commercial Success Metrics\n"
            "(Spearman correlation; * p<0.05, ** p<0.01, *** p<0.001)",
            fontsize=12,
            pad=14,
        )
        ax.set_xlabel("Success Metric", fontsize=11)
        ax.set_ylabel("Sentiment Feature", fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        plt.setp(ax.get_yticklabels(), rotation=0)

        n_min = int(n_grid[n_grid > 0].min()) if (n_grid > 0).any() else 0
        n_max = int(n_grid.max())
        fig.text(
            0.99,
            0.01,
            f"n = {n_min}–{n_max} games per cell",
            ha="right",
            va="bottom",
            fontsize=8,
            color="gray",
        )

        plt.tight_layout()

        save_path = self.figures_dir / "feature_metric_grid.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_success_comparison(
        self,
        success_correlations: Dict,
        post_release_correlation: Optional["CorrelationResult"] = None,
    ) -> None:
        """Create bar chart comparing correlation strengths across metrics."""
        if not success_correlations and not post_release_correlation:
            print("No correlation results to plot")
            return

        data = []

        if post_release_correlation:
            data.append(
                {
                    "Metric": "Post-Release\nSentiment",
                    "Correlation": post_release_correlation.spearman_r,
                    "p-value": post_release_correlation.spearman_p,
                    "Significant": post_release_correlation.is_significant,
                }
            )

        for name, corr in success_correlations.items():
            display_name = name.replace(" (Positive Ratio)", "\n(Ratio)")
            display_name = display_name.replace("Estimated Sales", "Est.\nSales")
            display_name = display_name.replace("Average Playtime", "Avg\nPlaytime")
            display_name = display_name.replace("Metacritic Score", "Metacritic")
            display_name = display_name.replace("Review Score", "Review\nScore")

            data.append(
                {
                    "Metric": display_name,
                    "Correlation": corr.spearman_r,
                    "p-value": corr.spearman_p,
                    "Significant": corr.is_significant,
                }
            )

        df = pd.DataFrame(data)

        df["AbsCorr"] = df["Correlation"].abs()
        df = df.sort_values("AbsCorr", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = []
        for _, row in df.iterrows():
            if not row["Significant"]:
                colors.append("#bdc3c7")  # Gray for non-significant
            elif row["Correlation"] > 0:
                colors.append("#27ae60")  # Green for positive
            else:
                colors.append("#e74c3c")  # Red for negative

        y_pos = range(len(df))
        bars = ax.barh(
            y_pos, df["Correlation"], color=colors, alpha=0.8, edgecolor="black"
        )

        for i, (idx, row) in enumerate(df.iterrows()):
            label = f"{row['Correlation']:.2f}"
            if row["Significant"]:
                label += " *"
            x_pos = row["Correlation"] + (0.02 if row["Correlation"] >= 0 else -0.02)
            ha = "left" if row["Correlation"] >= 0 else "right"
            ax.text(x_pos, i, label, va="center", ha=ha, fontsize=9, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Metric"], fontsize=10)
        ax.set_xlabel("Spearman Correlation (r)", fontsize=12)

        ax.axvline(x=0, color="black", linewidth=0.8)

        ax.axvline(
            x=0.3, color="#95a5a6", linestyle="--", alpha=0.5, label="Medium effect"
        )
        ax.axvline(x=-0.3, color="#95a5a6", linestyle="--", alpha=0.5)
        ax.axvline(
            x=0.5, color="#7f8c8d", linestyle="--", alpha=0.5, label="Large effect"
        )
        ax.axvline(x=-0.5, color="#7f8c8d", linestyle="--", alpha=0.5)

        ax.set_xlim(-1, 1)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        save_path = self.figures_dir / "success_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def generate_validation_plots(
        self,
        game_sentiment: pd.DataFrame,
        success_metrics: pd.DataFrame,
        success_correlations: Dict,
        post_release_correlation: Optional["CorrelationResult"] = None,
    ) -> None:
        """Generate all extended validation visualizations."""
        print("\n" + "=" * 60)
        print("GENERATING EXTENDED VALIDATION PLOTS")
        print("=" * 60)

        print("\n1. Success Correlations Scatter Plots")
        self.plot_success_correlations(
            game_sentiment, success_metrics, success_correlations
        )

        print("\n2. Correlation Matrix Heatmap")
        self.plot_correlation_matrix(game_sentiment, success_metrics)

        print("\n3. Feature x Metric Correlation Grid")
        self.plot_feature_metric_grid(game_sentiment, success_metrics)

        print("\n4. Success Comparison Bar Chart")
        self.plot_success_comparison(success_correlations, post_release_correlation)

        print(f"\nAll extended validation plots saved to: {self.figures_dir}")
