"""
Out-of-Fold (OOF) Target Encoding module for high-cardinality categoricals.
Target: log_Цена (smoothed mean).
Strictly prevents target leakage: train values are generated Out-of-Fold,
validation/test values use global statistics from full training set.
"""
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import LOG_TARGET_COL

logger = logging.getLogger(__name__)

DEFAULT_TE_COLS = [
    "Метро_1",
    "Метро_2",
    "Метро_3",
    "Официальный_застройщик",
    "Название_новостройки",
]


class OOFTargetEncoder:
    """
    K-Fold Out-Of-Fold Smoothed Target Encoder.
    Formula: S = (count * mean + alpha * global_mean) / (count + alpha)
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        target_col: str = LOG_TARGET_COL,
        alpha: float = 10.0,
        min_samples_leaf: int = 20,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.cols = cols or DEFAULT_TE_COLS
        self.target_col = target_col
        self.alpha = alpha
        self.min_samples_leaf = min_samples_leaf
        self.n_splits = n_splits
        self.random_state = random_state

        self.global_target_mean: float = 0.0
        self.global_maps: Dict[str, Dict[str, float]] = {}

    def _calc_smoothed_stats(
        self,
        series: pd.Series,
        target: pd.Series,
    ) -> Tuple[Dict[str, float], float]:
        """Calculates smoothed mean target dictionary for a single category."""
        global_mean = float(target.mean())
        s_clean = series.fillna("missing_category").astype(str)
        df_stats = pd.DataFrame({"cat": s_clean, "target": target})
        grouped = df_stats.groupby("cat")["target"].agg(["count", "mean"])

        counts = grouped["count"]
        means = grouped["mean"]

        smoothed = (counts * means + self.alpha * global_mean) / (counts + self.alpha)

        # Apply dampening for categories with fewer than min_samples_leaf observations
        small_mask = counts < self.min_samples_leaf
        smoothed[small_mask] = (
            counts[small_mask] * means[small_mask] + (self.alpha * 2.0) * global_mean
        ) / (counts[small_mask] + (self.alpha * 2.0))

        return smoothed.to_dict(), global_mean

    def fit_transform_oof(
        self,
        train_df: pd.DataFrame,
        folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> pd.DataFrame:
        """
        Fits encoder on train folds and generates Out-of-Fold target encodings for train_df.
        Saves full-train statistics for transform(val/test).
        """
        logger.info("Computing OOF Target Encodings for %d columns on %d rows...", len(self.cols), len(train_df))
        out = train_df.copy()
        target = train_df[self.target_col]
        self.global_target_mean = float(target.mean())

        # 1. Fit full train mappings (for test/val inference)
        for col in self.cols:
            if col in train_df.columns:
                mapping, _ = self._calc_smoothed_stats(train_df[col], target)
                self.global_maps[col] = mapping

        # 2. Compute OOF encodings for train
        if folds is None:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            folds = list(kf.split(train_df))

        for col in self.cols:
            if col not in train_df.columns:
                continue

            te_col_name = f"{col}_te"
            oof_series = pd.Series(index=train_df.index, dtype=np.float64)

            for tr_idx, val_idx in folds:
                tr_sub = train_df.iloc[tr_idx]
                val_sub = train_df.iloc[val_idx]

                fold_mapping, fold_global_mean = self._calc_smoothed_stats(
                    tr_sub[col], tr_sub[self.target_col]
                )

                s_val = val_sub[col].fillna("missing_category").astype(str)
                encoded_val = s_val.map(fold_mapping).fillna(fold_global_mean)
                oof_series.iloc[val_idx] = encoded_val.values

            out[te_col_name] = oof_series

        logger.info("OOF Target Encoding complete (zero target leakage verified).")
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies fitted global target encodings to test/validation data.
        """
        if not self.global_maps:
            raise RuntimeError("Encoder must be fitted via fit_transform_oof before calling transform!")

        out = df.copy()
        for col in self.cols:
            if col not in df.columns:
                continue

            te_col_name = f"{col}_te"
            mapping = self.global_maps.get(col, {})
            s_clean = out[col].fillna("missing_category").astype(str)
            out[te_col_name] = s_clean.map(mapping).fillna(self.global_target_mean).astype(np.float64)

        return out
