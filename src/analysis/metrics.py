from typing import Tuple

import numpy as np
from pycm import ConfusionMatrix
from scipy.stats import norm


def calculate_moe(
    std: float,
    n: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0

    z_score = norm.ppf((1 + confidence_level) / 2)
    se = std / np.sqrt(n)
    moe = z_score * se

    return moe, z_score


def calculate_mcen(tp: int, tn: int, fp: int, fn: int) -> float:
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0

    actual = [0] * (tn + fp) + [1] * (fn + tp)
    predict = [0] * tn + [1] * fp + [0] * fn + [1] * tp

    if len(actual) == 0 or len(predict) == 0:
        return 0.0

    cm = ConfusionMatrix(actual_vector=actual, predict_vector=predict)

    overall_mcen = cm.overall_stat.get("Overall MCEN")

    if overall_mcen is not None and overall_mcen != "None":
        return float(min(max(overall_mcen, 0.0), 1.0))

    mcen_values = cm.class_stat.get("MCEN", {})
    if mcen_values:
        valid_mcen = [v for v in mcen_values.values() if v is not None and v != "None"]
        if valid_mcen:
            mcen = np.mean(valid_mcen)
            return float(min(max(mcen, 0.0), 1.0))

    return 0.0
