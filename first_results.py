import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def create_results_dataframe():
    data = {
        "ELECTRA": {
            "f1": 0.8894,
            "std_f1": 0.0041,
            "accuracy": 0.8968,
            "precision": 0.8924,
            "recall": 0.8968,
            "time": 162.28,
            "type": "Transformer",
        },
        "RoBERTa": {
            "f1": 0.8810,
            "std_f1": 0.0100,
            "accuracy": 0.8906,
            "precision": 0.8830,
            "recall": 0.8906,
            "time": 165.08,
            "type": "Transformer",
        },
        "Gradient Boosting": {
            "f1": 0.7871,
            "std_f1": 0.0065,
            "accuracy": 0.8351,
            "precision": 0.7931,
            "recall": 0.8351,
            "time": 2.37,
            "type": "Classical",
        },
        "XGBoost": {
            "f1": 0.7719,
            "std_f1": 0.0074,
            "accuracy": 0.8215,
            "precision": 0.7574,
            "recall": 0.8215,
            "time": 0.32,
            "type": "Classical",
        },
        "LightGBM": {
            "f1": 0.7709,
            "std_f1": 0.0063,
            "accuracy": 0.8239,
            "precision": 0.7558,
            "recall": 0.8239,
            "time": 0.24,
            "type": "Classical",
        },
        "SVM": {
            "f1": 0.7651,
            "std_f1": 0.0016,
            "accuracy": 0.8371,
            "precision": 0.7505,
            "recall": 0.8371,
            "time": 0.58,
            "type": "Classical",
        },
        "Logistic Regression": {
            "f1": 0.7613,
            "std_f1": 0.0000,
            "accuracy": 0.8360,
            "precision": 0.6989,
            "recall": 0.8360,
            "time": 0.01,
            "type": "Classical",
        },
        "Naive Bayes": {
            "f1": 0.7613,
            "std_f1": 0.0000,
            "accuracy": 0.8360,
            "precision": 0.6989,
            "recall": 0.8360,
            "time": 0.001,
            "type": "Classical",
        },
    }

    df = pd.DataFrame(data).T
    df.index.name = "Model"
    return df


def create_clean_table_image(df):
    """Create a clean table as an image with proper title."""

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")

    sorted_df = df.sort_values("f1", ascending=False)

    table_data = []
    for rank, (model, row) in enumerate(sorted_df.iterrows(), 1):
        table_data.append(
            [
                f"{rank}",
                model,
                row["type"],
                f"{row['f1']:.3f}",
                f"±{row['std_f1']:.3f}",
                f"{row['accuracy']:.3f}",
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['time']:.2f}s" if row["time"] >= 0.01 else f"{
                    row['time']:.3f}s",
            ]
        )

    headers = [
        "#",
        "Model",
        "Type",
        "Avg F1",
        "Std Dev",
        "Avg Acc",
        "Avg Prec",
        "Avg Recall",
        "Avg Time",
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 0.85],
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    # Header row styling
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(weight="bold", color="white", fontsize=12)
        cell.set_edgecolor("white")
        cell.set_linewidth(2)

    for i in range(1, len(table_data) + 1):
        if i <= 2:
            bg_color = "#fff3cd"
            text_weight = "bold"
        else:
            bg_color = "#f8f9fa" if i % 2 == 0 else "white"
            text_weight = "normal"

        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor(bg_color)
            cell.set_text_props(weight=text_weight, fontsize=10)
            cell.set_edgecolor("#dee2e6")
            cell.set_linewidth(1)

    fig.text(
        0.5,
        0.95,
        "Sentiment Analysis Model Comparison",
        ha="center",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.91,
        "Average results from 60 runs (12 seeds × 5 folds per model) Total 480 runs",
        ha="center",
        fontsize=12,
        style="italic",
        color="#555",
    )

    plt.tight_layout()
    plt.savefig(
        "comparison_table_final.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print("✓ Table saved: comparison_table_final.png")


def create_simple_bar_chart(df):
    """Create simple bar chart - NO STD DEV TEXT IN LABELS."""

    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_df = df.sort_values("f1", ascending=True)

    colors = ["#e74c3c" if t ==
              "Transformer" else "#3498db" for t in sorted_df["type"]]

    y_pos = np.arange(len(sorted_df))

    bars = ax.barh(y_pos, sorted_df["f1"], color=colors, height=0.7, alpha=0.9)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df.index, fontsize=11)
    ax.set_xlabel("Average F1-Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Performance Comparison\nAverage F1-Score from 60 runs (12 seeds × 5 folds)Total 480 runs",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    for i, val in enumerate(sorted_df["f1"]):
        ax.text(
            val + 0.003, i, f"{val:.3f}", va="center", fontsize=10, fontweight="bold"
        )

    trans_patch = mpatches.Patch(color="#e74c3c", label="Transformer")
    class_patch = mpatches.Patch(color="#3498db", label="Classical ML")
    ax.legend(handles=[trans_patch, class_patch],
              loc="lower right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_xlim([0.70, 0.92])

    plt.tight_layout()
    plt.savefig(
        "simple_f1_chart_final.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Chart saved: simple_f1_chart_final.png")


def main():
    """Main function."""

    df = create_results_dataframe()

    print("\nCreating final clean visualizations...\n")

    create_clean_table_image(df)
    create_simple_bar_chart(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
