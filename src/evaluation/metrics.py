"""
Evaluation metrics module for house prices prediction.
All percentage and currency metrics are calculated on original RUB scale (expm1),
plus R2_log and RMSLE calculated on log1p scale.
"""
from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true_rub: np.ndarray, y_pred_rub: np.ndarray, y_true_log: np.ndarray = None, y_pred_log: np.ndarray = None) -> Dict[str, float]:
    """
    Computes all 7 standard Gate 3 metrics + supplementary stats:
    1. MdAPE (%): Median Absolute Percentage Error
    2. MAPE (%): Mean Absolute Percentage Error
    3. WAPE (%): Weighted Absolute Percentage Error
    4. RMSLE: Root Mean Squared Logarithmic Error
    5. PEP10 (%): Predictions within +-10% error
    6. PEP20 (%): Predictions within +-20% error (IAAO standard)
    7. R2_log: R^2 on log scale
    Supplementary: MAE (rub), RMSE (rub), R2 (rub)
    """
    y_true = np.asarray(y_true_rub, dtype=np.float64)
    y_pred = np.asarray(y_pred_rub, dtype=np.float64)

    y_pred = np.clip(y_pred, a_min=1.0, a_max=None)
    y_true_safe = np.clip(y_true, a_min=1.0, a_max=None)

    abs_err = np.abs(y_true - y_pred)
    pct_err = abs_err / y_true_safe

    mdape = float(np.median(pct_err) * 100.0)
    mape = float(np.mean(pct_err) * 100.0)
    wape = float((np.sum(abs_err) / np.sum(y_true_safe)) * 100.0)
    pep10 = float(np.mean(pct_err <= 0.10) * 100.0)
    pep20 = float(np.mean(pct_err <= 0.20) * 100.0)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_rub = float(r2_score(y_true, y_pred))

    if y_true_log is None:
        y_true_log = np.log1p(y_true_safe)
    if y_pred_log is None:
        y_pred_log = np.log1p(y_pred)

    rmsle = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    r2_log = float(r2_score(y_true_log, y_pred_log))

    return {
        "mdape": mdape,
        "mape": mape,
        "wape": wape,
        "rmsle": rmsle,
        "pep10": pep10,
        "pep20": pep20,
        "r2_log": r2_log,
        "mae": mae,
        "rmse": rmse,
        "r2_rub": r2_rub,
    }


def calculate_metrics_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """Computes all metrics given log1p predictions and targets."""
    y_true_rub = np.expm1(y_true_log)
    y_pred_rub = np.expm1(y_pred_log)
    return calculate_metrics(y_true_rub, y_pred_rub, y_true_log, y_pred_log)
