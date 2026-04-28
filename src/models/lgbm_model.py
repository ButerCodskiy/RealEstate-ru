"""
LightGBM regression model and Optuna hyperparameter optimization.
"""
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.evaluation.metrics import calculate_metrics_from_log

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_default_lgbm_params() -> Dict[str, Any]:
    return {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }


def fit_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMRegressor:
    """Trains LightGBM model with optional early stopping."""
    p = params or get_default_lgbm_params()
    model = lgb.LGBMRegressor(**p)

    if X_val is not None and y_val is not None:
        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
    else:
        model.fit(X_train, y_train)

    return model


def tune_lgbm(
    X: np.ndarray,
    y_log: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 60,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float, optuna.Study]:
    """
    Optuna hyperparameter tuning for LightGBM.
    Target metric to minimize: MdAPE on Out-Of-Fold predictions.
    """
    logger.info("Starting Optuna study for LightGBM (%d trials)...", n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": seed,
            "verbose": -1,
            "n_jobs": -1,
        }

        # Evaluate across folds
        val_errors = []
        # Use first 3 folds during optimization for 2x faster search
        eval_folds = folds[:3] if len(folds) >= 3 else folds
        for tr_idx, val_idx in eval_folds:
            X_tr, y_tr = X[tr_idx], y_log[tr_idx]
            X_val, y_val = X[val_idx], y_log[val_idx]

            m = fit_lgbm(X_tr, y_tr, X_val, y_val, params=params, early_stopping_rounds=30)
            preds = m.predict(X_val)
            metrics = calculate_metrics_from_log(y_val, preds)
            val_errors.append(metrics["mdape"])

        mean_mdape = float(np.mean(val_errors))
        return mean_mdape

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = get_default_lgbm_params()
    best_params.update(study.best_params)
    best_score = study.best_value

    logger.info("LightGBM Optuna finished. Best Trial MdAPE: %.2f%%", best_score)
    return best_params, best_score, study
