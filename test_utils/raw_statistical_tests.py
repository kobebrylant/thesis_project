from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp


def run_friedman_test_raw(data: np.ndarray) -> Dict:
    if data.ndim != 2:
        raise ValueError("Data must be 2D array (subjects x treatments)")

    n_subjects, n_treatments = data.shape

    data_cols = [data[:, i] for i in range(n_treatments)]
    stat, p_value = friedmanchisquare(*data_cols)

    kendall_w = stat / (n_subjects * (n_treatments - 1))

    return {
        "chi2": float(stat),
        "p_value": float(p_value),
        "kendall_w": float(kendall_w),
        "n_subjects": n_subjects,
        "n_treatments": n_treatments,
        "df": n_treatments - 1,
    }


def run_wilcoxon_test_raw(x: np.ndarray, y: np.ndarray) -> Dict:
    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    stat, p_value = wilcoxon(x, y, alternative="two-sided")

    n = len(x)
    max_rank_sum = n * (n + 1) / 2
    r = 1 - (2 * stat) / max_rank_sum

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "effect_size_r": float(r),
        "n_pairs": n,
        "mean_diff": float(np.mean(y - x)),
    }


def run_nemenyi_test_raw(
    data: np.ndarray, labels: Optional[List[str]] = None
) -> pd.DataFrame:
    if data.ndim != 2:
        raise ValueError("Data must be 2D array (subjects x treatments)")

    nemenyi_results = sp.posthoc_nemenyi_friedman(data)

    if labels is not None:
        nemenyi_results.index = labels
        nemenyi_results.columns = labels

    return nemenyi_results
