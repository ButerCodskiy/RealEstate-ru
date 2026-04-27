"""
RandomForest baseline model for comparison in tournament.
"""
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def get_default_rf_params() -> Dict[str, Any]:
    return {
        "n_estimators": 120,
        "max_depth": 16,
        "min_samples_split": 6,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }


def fit_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> RandomForestRegressor:
    """Fits RandomForest baseline model."""
    p = params or get_default_rf_params()
    model = RandomForestRegressor(**p)
    model.fit(X_train, y_train)
    return model
