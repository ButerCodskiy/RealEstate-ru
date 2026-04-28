"""
CatBoost regression model and Optuna hyperparameter optimization.
"""
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.evaluation.metrics import calculate_metrics_from_log

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_default_catboost_params() -> Dict[str, Any]:
    return {
        "iterations": 800,
        "learning_rate": 0.06,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
    }


def fit_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 40,
) -> CatBoostRegressor:
    """Trains CatBoost model with early stopping."""
    p = params or get_default_catboost_params()
    model = CatBoostRegressor(**p)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping_rounds,
            verbose=0,
        )
    else:
        model.fit(X_train, y_train, verbose=0)

    return model


def tune_catboost(
    X: np.ndarray,
    y_log: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 50,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float, optuna.Study]:
    """
    Optuna hyperparameter tuning for CatBoost.
    Target metric to minimize: MdAPE on Out-Of-Fold predictions.
    """
    logger.info("Starting Optuna study for CatBoost (%d trials)...", n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "iterations": trial.suggest_int("iterations", 300, 1000, step=100),
            "depth": trial.suggest_int("depth", 5, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
            "random_seed": seed,
            "verbose": 0,
            "allow_writing_files": False,
        }

        val_errors = []
        eval_folds = folds[:3] if len(folds) >= 3 else folds
        for tr_idx, val_idx in eval_folds:
            X_tr, y_tr = X[tr_idx], y_log[tr_idx]
            X_val, y_val = X[val_idx], y_log[val_idx]

            m = fit_catboost(X_tr, y_tr, X_val, y_val, params=params, early_stopping_rounds=25)
            preds = m.predict(X_val)
            metrics = calculate_metrics_from_log(y_val, preds)
            val_errors.append(metrics["mdape"])

        return float(np.mean(val_errors))

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=8)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = get_default_catboost_params()
    best_params.update(study.best_params)
    best_score = study.best_value

    logger.info("CatBoost Optuna finished. Best Trial MdAPE: %.2f%%", best_score)
    return best_params, best_score, study
