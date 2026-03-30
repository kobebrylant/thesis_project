#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp

TEST_DATA_DIR = Path("test_data_verification")


def generate_mock_friedman_data(
    n_subjects: int = 10,
    n_treatments: int = 4,
    seed: int = 42,
    effect_strength: str = "moderate",
) -> pd.DataFrame:
    np.random.seed(seed)

    if effect_strength == "none":
        treatment_means = [50] * n_treatments
    elif effect_strength == "weak":
        treatment_means = [50 + i * 2 for i in range(n_treatments)]
    elif effect_strength == "moderate":
        treatment_means = [50 + i * 5 for i in range(n_treatments)]
    else:
        treatment_means = [50 + i * 10 for i in range(n_treatments)]

    data = {"subject": list(range(1, n_subjects + 1))}
    subject_effects = np.random.normal(0, 10, n_subjects)

    for i, mean in enumerate(treatment_means, 1):
        scores = mean + subject_effects + np.random.normal(0, 5, n_subjects)
        scores = np.clip(scores, 0, 100)
        data[f"treatment_{i}"] = np.round(scores, 2)

    return pd.DataFrame(data)


def generate_mock_wilcoxon_data(
    n_pairs: int = 20,
    seed: int = 42,
    effect_type: str = "moderate",
) -> pd.DataFrame:
    np.random.seed(seed)

    effect_sizes = {
        "none": 0,
        "weak": 2,
        "moderate": 5,
        "strong": 10,
    }
    effect = effect_sizes.get(effect_type, 5)

    condition_a = np.random.normal(50, 10, n_pairs)
    condition_b = condition_a + effect + np.random.normal(0, 5, n_pairs)

    return pd.DataFrame({
        "pair_id": list(range(1, n_pairs + 1)),
        "condition_a": np.round(condition_a, 2),
        "condition_b": np.round(condition_b, 2),
        "difference": np.round(condition_b - condition_a, 2),
    })


def generate_thesis_format_data(
    n_seeds: int = 5,
    n_folds: int = 5,
    n_models: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)

    model_names = [f"Model_{chr(65 + i)}" for i in range(n_models)]
    model_types = ["classical"] * (n_models // 2) + ["transformer"] * (n_models - n_models // 2)
    base_f1 = {model_names[i]: 0.7 + i * 0.05 for i in range(n_models)}

    rows = []
    for seed_idx in range(n_seeds):
        for fold_idx in range(n_folds):
            for model_idx, model in enumerate(model_names):
                f1 = base_f1[model] + np.random.normal(0, 0.02)
                f1 = np.clip(f1, 0, 1)

                accuracy = f1 + np.random.normal(0, 0.01)
                precision = f1 + np.random.normal(0.02, 0.02)
                recall = f1 + np.random.normal(-0.02, 0.02)

                rows.append({
                    "seed": seed_idx,
                    "fold": fold_idx,
                    "model": model,
                    "type": model_types[model_idx],
                    "f1_score": round(np.clip(f1, 0, 1), 4),
                    "accuracy": round(np.clip(accuracy, 0, 1), 4),
                    "precision": round(np.clip(precision, 0, 1), 4),
                    "recall": round(np.clip(recall, 0, 1), 4),
                    "train_time": round(np.random.uniform(10, 100), 2),
                    "inference_time_ms": round(np.random.uniform(1, 10), 4),
                })

    return pd.DataFrame(rows)


def export_for_pspp(df: pd.DataFrame, filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{filename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    tsv_path = output_dir / f"{filename}.tsv"
    df.to_csv(tsv_path, index=False, sep="\t")
    print(f"  TSV: {tsv_path}")


def generate_pspp_syntax_friedman(
    filename: str,
    treatment_cols: list,
    output_dir: Path,
) -> None:
    abs_csv_path = (output_dir / f"{filename}.csv").absolute()
    syntax = f"""* PSPP Syntax for Friedman Test Verification

* Import data
GET DATA /TYPE=TXT
  /FILE='{abs_csv_path}'
  /ARRANGEMENT=DELIMITED
  /DELIMITERS=","
  /FIRSTCASE=2
  /VARIABLES=
    subject F8.0
    {chr(10).join(f'    {col} F8.2' for col in treatment_cols)}.

* Run Friedman test
NPAR TESTS
  /FRIEDMAN = {' '.join(treatment_cols)}.

* Show descriptives
DESCRIPTIVES VARIABLES={' '.join(treatment_cols)}
  /STATISTICS=MEAN STDDEV MIN MAX.
"""

    syntax_path = output_dir / f"{filename}_friedman.sps"
    with open(syntax_path, "w") as f:
        f.write(syntax)
    print(f"  PSPP syntax: {syntax_path}")


def generate_pspp_syntax_wilcoxon(
    filename: str,
    output_dir: Path,
) -> None:
    abs_csv_path = (output_dir / f"{filename}.csv").absolute()
    syntax = f"""* PSPP Syntax for Wilcoxon Signed-Rank Test Verification

* Import data
GET DATA /TYPE=TXT
  /FILE='{abs_csv_path}'
  /ARRANGEMENT=DELIMITED
  /DELIMITERS=","
  /FIRSTCASE=2
  /VARIABLES=
    pair_id F8.0
    condition_a F8.2
    condition_b F8.2
    difference F8.2.

* Run Wilcoxon signed-rank test
NPAR TESTS
  /WILCOXON = condition_a WITH condition_b (PAIRED).

* Show descriptives
DESCRIPTIVES VARIABLES=condition_a condition_b difference
  /STATISTICS=MEAN STDDEV MIN MAX.
"""

    syntax_path = output_dir / f"{filename}_wilcoxon.sps"
    with open(syntax_path, "w") as f:
        f.write(syntax)
    print(f"  PSPP syntax: {syntax_path}")


def run_python_friedman_test(df: pd.DataFrame, treatment_cols: list) -> dict:
    data = [df[col].values for col in treatment_cols]
    stat, p_value = friedmanchisquare(*data)

    n = len(df)
    k = len(treatment_cols)
    kendall_w = stat / (n * (k - 1))

    return {
        "test": "Friedman",
        "chi2": round(stat, 4),
        "p_value": p_value,
        "kendall_w": round(kendall_w, 4),
        "n_subjects": n,
        "n_treatments": k,
        "df": k - 1,
    }


def run_python_wilcoxon_test(df: pd.DataFrame) -> dict:
    x = df["condition_a"].values
    y = df["condition_b"].values

    stat, p_value = wilcoxon(x, y, alternative="two-sided")

    n = len(x)
    max_rank_sum = n * (n + 1) / 2
    r = 1 - (2 * stat) / max_rank_sum

    return {
        "test": "Wilcoxon",
        "statistic": stat,
        "p_value": p_value,
        "effect_size_r": round(r, 4),
        "n_pairs": n,
        "mean_a": round(np.mean(x), 4),
        "mean_b": round(np.mean(y), 4),
        "mean_diff": round(np.mean(y - x), 4),
    }


def run_python_nemenyi_test(df: pd.DataFrame, treatment_cols: list) -> pd.DataFrame:
    data_matrix = df[treatment_cols].values
    nemenyi_results = sp.posthoc_nemenyi_friedman(data_matrix)
    nemenyi_results.index = treatment_cols
    nemenyi_results.columns = treatment_cols
    return nemenyi_results


def run_thesis_implementation_tests(df: pd.DataFrame) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.config import PathConfig
    from src.analysis.statistical_tests import StatisticalAnalyzer

    path_config = PathConfig(
        base_dir=TEST_DATA_DIR,
        results_dir=TEST_DATA_DIR / "thesis_results",
    )
    analyzer = StatisticalAnalyzer(path_config)

    friedman_result, pivot_df, excluded = analyzer.friedman_test(df)
    wilcoxon_df, _ = analyzer.wilcoxon_pairwise(df)
    nemenyi_df = analyzer.nemenyi_posthoc(pivot_df)

    return {
        "friedman": {
            "chi2": round(friedman_result.chi2, 4),
            "p_value": friedman_result.p_value,
            "kendall_w": round(friedman_result.kendall_w, 4),
            "n_samples": friedman_result.n_samples,
            "n_models": friedman_result.n_models,
        },
        "wilcoxon": wilcoxon_df,
        "nemenyi": nemenyi_df,
        "pivot_df": pivot_df,
    }


def generate_all_test_data():
    print("\n" + "=" * 60)
    print("GENERATING TEST DATA FOR PSPP VERIFICATION")
    print("=" * 60)

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Friedman Test Data")
    print("-" * 40)

    for effect in ["none", "weak", "moderate", "strong"]:
        print(f"\n  Effect: {effect}")
        df = generate_mock_friedman_data(
            n_subjects=10,
            n_treatments=4,
            seed=42,
            effect_strength=effect,
        )
        filename = f"friedman_{effect}"
        export_for_pspp(df, filename, TEST_DATA_DIR)
        treatment_cols = [c for c in df.columns if c.startswith("treatment_")]
        generate_pspp_syntax_friedman(filename, treatment_cols, TEST_DATA_DIR)

    print("\n2. Wilcoxon Test Data")
    print("-" * 40)

    for effect in ["none", "weak", "moderate", "strong"]:
        print(f"\n  Effect: {effect}")
        df = generate_mock_wilcoxon_data(
            n_pairs=20,
            seed=42,
            effect_type=effect,
        )
        filename = f"wilcoxon_{effect}"
        export_for_pspp(df, filename, TEST_DATA_DIR)
        generate_pspp_syntax_wilcoxon(filename, TEST_DATA_DIR)

    print("\n3. Thesis Format Data")
    print("-" * 40)
    df = generate_thesis_format_data(n_seeds=5, n_folds=5, n_models=4, seed=42)
    export_for_pspp(df, "thesis_format", TEST_DATA_DIR)

    df_copy = df.copy()
    df_copy["run_id"] = df_copy["seed"].astype(str) + "_" + df_copy["fold"].astype(str)
    pivot = df_copy.pivot_table(index="run_id", columns="model", values="f1_score", aggfunc="first")
    pivot = pivot.reset_index()
    export_for_pspp(pivot, "thesis_format_pivot", TEST_DATA_DIR)

    model_cols = [c for c in pivot.columns if c != "run_id"]
    generate_pspp_syntax_friedman("thesis_format_pivot", model_cols, TEST_DATA_DIR)

    print(f"\n\nAll test data saved to: {TEST_DATA_DIR.absolute()}")
    print("\nInstructions for PSPP verification:")
    print("1. Open PSPP")
    print("2. File -> Open -> Syntax")
    print("3. Select a .sps file from the test_data_verification directory")
    print("4. Run -> All")
    print("5. Compare results with Python output")


def run_all_tests():
    print("\n" + "=" * 60)
    print("PYTHON STATISTICAL TEST RESULTS")
    print("=" * 60)

    results_summary = []

    print("\n1. FRIEDMAN TEST RESULTS")
    print("-" * 40)

    for effect in ["none", "weak", "moderate", "strong"]:
        df = generate_mock_friedman_data(
            n_subjects=10,
            n_treatments=4,
            seed=42,
            effect_strength=effect,
        )
        treatment_cols = [c for c in df.columns if c.startswith("treatment_")]
        result = run_python_friedman_test(df, treatment_cols)

        print(f"\n  Effect: {effect}")
        print(f"  " + "=" * 50)
        print(f"  Descriptive Statistics (compare with PSPP):")
        for col in treatment_cols:
            print(f"    {col}  Mean: {df[col].mean():.2f}  Std: {df[col].std():.2f}")
        print(f"    N: {result['n_subjects']}")
        print(f"  " + "-" * 50)
        print(f"  Test Statistics (compare with PSPP):")
        print(f"    Chi-squared: {result['chi2']:.4f}")
        print(f"    df: {result['df']}")
        print(f"    p-value (Asymp. Sig.): {result['p_value']:.6f}")
        print(f"  " + "-" * 50)
        print(f"  Additional (Python only):")
        print(f"    Kendall's W: {result['kendall_w']:.4f}")

        results_summary.append({
            "test": f"Friedman_{effect}",
            "statistic": result["chi2"],
            "p_value": result["p_value"],
            "effect_size": result["kendall_w"],
        })

        nemenyi_df = run_python_nemenyi_test(df, treatment_cols)
        print(f"\n    Nemenyi post-hoc p-values:")
        print(nemenyi_df.round(4).to_string().replace("\n", "\n    "))

    print("\n\n2. WILCOXON SIGNED-RANK TEST RESULTS")
    print("-" * 40)

    for effect in ["none", "weak", "moderate", "strong"]:
        df = generate_mock_wilcoxon_data(
            n_pairs=20,
            seed=42,
            effect_type=effect,
        )
        result = run_python_wilcoxon_test(df)

        print(f"\n  Effect: {effect}")
        print(f"  " + "=" * 50)
        print(f"  Descriptive Statistics (compare with PSPP):")
        print(f"    condition_a  Mean: {result['mean_a']:.2f}")
        print(f"    condition_b  Mean: {result['mean_b']:.2f}")
        print(f"    difference   Mean: {result['mean_diff']:.2f}")
        print(f"    N: {result['n_pairs']}")
        print(f"  " + "-" * 50)
        print(f"  Test Statistics (compare with PSPP):")
        print(f"    W statistic: {result['statistic']}")
        print(f"    p-value (Asymp. Sig. 2-tailed): {result['p_value']:.6f}")
        print(f"  " + "-" * 50)
        print(f"  Additional (Python only):")
        print(f"    Effect size (r): {result['effect_size_r']}")

        results_summary.append({
            "test": f"Wilcoxon_{effect}",
            "statistic": result["statistic"],
            "p_value": result["p_value"],
            "effect_size": result["effect_size_r"],
        })

    print("\n\n3. THESIS IMPLEMENTATION TEST RESULTS")
    print("-" * 40)

    df = generate_thesis_format_data(n_seeds=5, n_folds=5, n_models=4, seed=42)
    thesis_results = run_thesis_implementation_tests(df)

    print("\n  Friedman Test:")
    for key, value in thesis_results["friedman"].items():
        print(f"    {key}: {value}")

    print("\n  Nemenyi post-hoc p-values:")
    print(thesis_results["nemenyi"].round(4).to_string().replace("\n", "\n    "))

    print("\n  Wilcoxon pairwise comparisons:")
    wilcoxon_display = thesis_results["wilcoxon"][
        ["Model 1", "Model 2", "Statistic", "p-value", "Effect Size (r)"]
    ]
    print(wilcoxon_display.to_string(index=False).replace("\n", "\n    "))

    summary_df = pd.DataFrame(results_summary)
    summary_path = TEST_DATA_DIR / "python_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n\nResults summary saved to: {summary_path}")

    detailed_path = TEST_DATA_DIR / "python_detailed_results.txt"
    with open(detailed_path, "w") as f:
        f.write("PYTHON STATISTICAL TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write("These results should match PSPP output.\n\n")

        f.write("FRIEDMAN TEST RESULTS\n")
        f.write("-" * 40 + "\n")
        for effect in ["none", "weak", "moderate", "strong"]:
            df = generate_mock_friedman_data(n_subjects=10, n_treatments=4, seed=42, effect_strength=effect)
            treatment_cols = [c for c in df.columns if c.startswith("treatment_")]
            result = run_python_friedman_test(df, treatment_cols)
            f.write(f"\nEffect: {effect}\n")
            f.write(f"  Chi-squared: {result['chi2']}\n")
            f.write(f"  p-value: {result['p_value']:.10f}\n")
            f.write(f"  Kendall's W: {result['kendall_w']}\n")

        f.write("\n\nWILCOXON SIGNED-RANK TEST RESULTS\n")
        f.write("-" * 40 + "\n")
        for effect in ["none", "weak", "moderate", "strong"]:
            df = generate_mock_wilcoxon_data(n_pairs=20, seed=42, effect_type=effect)
            result = run_python_wilcoxon_test(df)
            f.write(f"\nEffect: {effect}\n")
            f.write(f"  Statistic (W): {result['statistic']}\n")
            f.write(f"  p-value: {result['p_value']:.10f}\n")
            f.write(f"  Effect size (r): {result['effect_size_r']}\n")

    print(f"Detailed results saved to: {detailed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical test verification against PSPP"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate mock test data and PSPP syntax files",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run statistical tests on mock data",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate data and run tests",
    )

    args = parser.parse_args()

    if not any([args.generate, args.run_tests, args.all]):
        parser.print_help()
        return

    if args.generate or args.all:
        generate_all_test_data()

    if args.run_tests or args.all:
        run_all_tests()


if __name__ == "__main__":
    main()
