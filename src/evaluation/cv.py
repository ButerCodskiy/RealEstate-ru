"""
Cross-validation and Out-Of-Fold (OOF) evaluation framework.
"""
from typing import Callable, List, Tuple, Dict, Any
import logging
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_metrics_from_log

logger = logging.getLogger(__name__)


def evaluate_model_oof(
    train_model_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any],
    X: np.ndarray,
    y_log: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, float], np.ndarray, List[Any]]:
    """
    Standardized Out-Of-Fold evaluation runner:
    - Iterates over cross-validation folds
    - Calls train_model_fn(X_train, y_train, X_val, y_val) -> fitted_model
    - Collects OOF predictions across all training samples
    - Computes all 7 metrics on the complete OOF vector
    
    Returns: (oof_metrics, oof_predictions, fitted_models)
    """
    n_samples = len(X)
    oof_preds = np.zeros(n_samples, dtype=np.float64)
    fitted_models = []

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X[tr_idx], y_log[tr_idx]
        X_val, y_val = X[val_idx], y_log[val_idx]

        model = train_model_fn(X_tr, y_tr, X_val, y_val)
        val_pred = model.predict(X_val)

        oof_preds[val_idx] = val_pred
        fitted_models.append(model)

    oof_metrics = calculate_metrics_from_log(y_log, oof_preds)
    return oof_metrics, oof_preds, fitted_models
