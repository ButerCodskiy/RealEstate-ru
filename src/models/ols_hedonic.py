"""
Hedonic OLS price specification, VIF calculation, and spatial autocorrelation (Moran's I).
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import KDTree
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import LOG_TARGET_COL


def prepare_ols_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prepares design matrix X for Hedonic OLS model strictly following PLAN.md specification:
    ln(Price) = alpha
      + beta1 * ln(Area)
      + beta2 * Rooms
      + beta3 * living_ratio
      + beta4 * floor_ratio
      + beta5 * is_first_floor
      + beta6 * is_top_floor
      + beta7 * ln(dist_to_center)
      + beta8 * metro_dist
      + beta9 * Repair (OHE)
      + beta10 * House_type (OHE)
      + beta11 * Bathroom (OHE)
      + beta12 * Ceiling_category (OHE)
      + beta13 * months_since_min
      + beta14 * is_new_building
      + gamma_k * geo_cluster (FE)
      + eps
    """
    data = df.copy()

    # 1. Rooms numeric parsing
    def _parse_rooms(val):
        if pd.isna(val):
            return 2.0
        val_str = str(val).strip().lower()
        if "студия" in val_str:
            return 0.7
        if "свободная" in val_str:
            return 1.0
        if "10" in val_str:
            return 10.0
        try:
            return float(val_str)
        except:
            return 2.0

    data["rooms_num"] = data["Количество_комнат"].apply(_parse_rooms)
    data["ln_area"] = np.log(np.maximum(data["Общая_площадь"], 1.0))
    data["ln_dist_center"] = np.log(np.maximum(data["dist_to_center_km"], 0.1))

    # 2. Metro dist numeric
    metro = data["Расстояние_до_метро_1"].copy()
    metro = metro.replace(1905.0, 15.0)  # repair known data error
    data["metro_dist_num"] = np.clip(metro.fillna(metro.median()), 0, 35)

    # 3. Categoricals cleanup
    data["house_type_clean"] = data["Тип_дома"].fillna("Не указан").replace("монолитнокирпичный", "монолитно-кирпичный")
    data["repair_clean"] = data["Ремонт"].fillna("Не указан")
    data["bathroom_clean"] = data["Санузел"].fillna("Не указан")
    data["ceiling_clean"] = data["ceiling_category"].fillna("standard_le_2.75")

    numeric_preds = [
        "ln_area", "rooms_num", "living_ratio", "floor_ratio",
        "is_first_floor", "is_top_floor", "ln_dist_center",
        "metro_dist_num", "months_since_min", "is_new_building"
    ]

    for col in numeric_preds:
        if data[col].isna().sum() > 0:
            data[col] = data[col].fillna(data[col].median())

    # 4. Dummy encoding (drop_first=True to avoid collinearity)
    cat_dummies = pd.get_dummies(
        data[["repair_clean", "house_type_clean", "bathroom_clean", "ceiling_clean"]],
        drop_first=True,
        dtype=float
    )

    cluster_dummies = pd.get_dummies(
        data["geo_cluster"].astype(str),
        prefix="cluster",
        drop_first=True,
        dtype=float
    )

    X = pd.concat([data[numeric_preds], cat_dummies, cluster_dummies], axis=1)
    X = sm.add_constant(X)
    y = data[LOG_TARGET_COL].values

    return X, y, numeric_preds


def calculate_vif(X_numeric: pd.DataFrame) -> pd.DataFrame:
    """Calculates Variance Inflation Factor for numeric continuous predictors."""
    X_const = sm.add_constant(X_numeric)
    vif_records = []
    for i, col in enumerate(X_numeric.columns):
        val = variance_inflation_factor(X_const.values, i + 1)
        vif_records.append({"feature": col, "VIF": float(val)})
    return pd.DataFrame(vif_records)


def fit_hedonic_ols(
    X: pd.DataFrame,
    y: np.ndarray,
    cov_type: str = "HC3",
    groups: np.ndarray = None
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fits statsmodels OLS model with robust HC3 or clustered standard errors."""
    model = sm.OLS(y, X)
    if cov_type == "cluster" and groups is not None:
        results = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        results = model.fit(cov_type=cov_type)
    return results


def compute_morans_i(coords: np.ndarray, residuals: np.ndarray, k: int = 8) -> Dict[str, float]:
    """
    Computes Moran's I spatial autocorrelation test using k-nearest neighbors spatial matrix.
    Fast O(N log N) implementation using KDTree and sparse row-standardized weights.
    """
    N = len(residuals)
    e = residuals - np.mean(residuals)
    tree = KDTree(coords)

    distances, indices = tree.query(coords, k=k + 1)

    row_idx = []
    col_idx = []
    weights = []

    for i in range(N):
        for j_pos in range(1, k + 1):
            j = indices[i, j_pos]
            d = distances[i, j_pos]
            w = 1.0 / max(d, 1e-5)
            row_idx.append(i)
            col_idx.append(j)
            weights.append(w)

    W = sp.csr_matrix((weights, (row_idx, col_idx)), shape=(N, N))
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W_norm = W.multiply(sp.diags(1.0 / row_sums))

    We = W_norm.dot(e)
    num = float(np.sum(e * We))
    denom = float(np.sum(e ** 2))
    I = num / denom

    E_I = -1.0 / (N - 1)
    S0 = float(W_norm.sum())
    W_sym = (W_norm + W_norm.T) / 2.0
    S1 = 2.0 * float(W_sym.multiply(W_sym).sum())

    w_row = np.array(W_norm.sum(axis=1)).flatten()
    w_col = np.array(W_norm.sum(axis=0)).flatten()
    S2 = float(np.sum((w_row + w_col) ** 2))

    var_I = (N * ((N**2 - 3*N + 3)*S1 - N*S2 + 3*S0**2)) / ((N - 1)*(N - 2)*(N - 3)*S0**2) - E_I**2
    std_I = np.sqrt(max(var_I, 1e-12))
    z_score = float((I - E_I) / std_I)
    p_val = float(2.0 * (1.0 - norm.cdf(abs(z_score))))

    return {
        "morans_i": I,
        "expected_i": E_I,
        "std_i": float(std_I),
        "z_score": z_score,
        "p_value": p_val,
        "k_neighbors": k,
        "n_samples": N,
        "is_significant": bool(p_val < 0.05),
    }
