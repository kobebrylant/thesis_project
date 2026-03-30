#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RESULTS_FILE = Path("results/checkpoints/training_results.csv")
OUTPUT_DIR = Path("test_data_verification/thesis_data")

def export_thesis_data_for_pspp():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        return
    
    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(df)} results")
    
    df["run_id"] = df["seed"].astype(str) + "_" + df["fold"].astype(str)
    
    # Pivot for Friedman test (each row = run, each column = model)
    pivot = df.pivot_table(
        index="run_id", 
        columns="model", 
        values="f1_score", 
        aggfunc="first"
    ).reset_index()
    
    pivot.to_csv(OUTPUT_DIR / "friedman_data.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'friedman_data.csv'}")
    
    # Generate PSPP syntax for Friedman
    model_cols = [c for c in pivot.columns if c != "run_id"]
    abs_path = (OUTPUT_DIR / "friedman_data.csv").absolute()
    
    syntax = f"""* PSPP Friedman Test for Thesis Data

GET DATA /TYPE=TXT
  /FILE='{abs_path}'
  /ARRANGEMENT=DELIMITED
  /DELIMITERS=","
  /FIRSTCASE=2
  /VARIABLES=
    run_id A10
    {chr(10).join(f'    {col.replace(" ", "_")} F8.4' for col in model_cols)}.

NPAR TESTS
  /FRIEDMAN = {' '.join(col.replace(" ", "_") for col in model_cols)}.

DESCRIPTIVES VARIABLES={' '.join(col.replace(" ", "_") for col in model_cols)}
  /STATISTICS=MEAN STDDEV MIN MAX.
"""
    
    with open(OUTPUT_DIR / "friedman_test.sps", "w") as f:
        f.write(syntax)
    print(f"Saved: {OUTPUT_DIR / 'friedman_test.sps'}")
    
    # Export pairwise data for Wilcoxon (example: best vs second best)
    models_sorted = pivot[model_cols].mean().sort_values(ascending=False)
    print(f"\nModel ranking by mean F1:")
    for i, (model, mean) in enumerate(models_sorted.items(), 1):
        print(f"  {i}. {model}: {mean:.4f}")
    
    # Create Wilcoxon syntax for each pair
    wilcoxon_syntax = f"""* PSPP Wilcoxon Tests for Thesis Data

GET DATA /TYPE=TXT
  /FILE='{abs_path}'
  /ARRANGEMENT=DELIMITED
  /DELIMITERS=","
  /FIRSTCASE=2
  /VARIABLES=
    run_id A10
    {chr(10).join(f'    {col.replace(" ", "_")} F8.4' for col in model_cols)}.

* Pairwise Wilcoxon tests
"""
    
    for i, model1 in enumerate(model_cols):
        for model2 in model_cols[i+1:]:
            m1 = model1.replace(" ", "_")
            m2 = model2.replace(" ", "_")
            wilcoxon_syntax += f"""
NPAR TESTS
  /WILCOXON = {m1} WITH {m2} (PAIRED).
"""
    
    with open(OUTPUT_DIR / "wilcoxon_tests.sps", "w") as f:
        f.write(wilcoxon_syntax)
    print(f"Saved: {OUTPUT_DIR / 'wilcoxon_tests.sps'}")
    
    print(f"\n" + "=" * 60)
    print("PSPP VERIFICATION INSTRUCTIONS")
    print("=" * 60)
    print(f"\n1. Open PSPP")
    print(f"2. File -> Open -> Syntax")
    print(f"3. Select: {OUTPUT_DIR.absolute()}/friedman_test.sps")
    print(f"4. Run -> All")
    print(f"\nCompare with Python results:")
    print(f"  - Chi-squared should match")
    print(f"  - p-value should match (PSPP shows as 'Asymp. Sig.')")
    print(f"  - df should be {len(model_cols) - 1}")

if __name__ == "__main__":
    export_thesis_data_for_pspp()
