"""
Data splitting module: Temporal split and Grouped K-Fold validation.
"""
from pathlib import Path
from typing import Tuple, List, Optional
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold

from src.config import (
    DATE_COL,
    TRAIN_END_DATE,
    VAL_END_DATE,
    DATA_INTERIM,
)

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END_DATE,
    val_end: str = VAL_END_DATE,
    save_interim: bool = True,
    interim_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs strict temporal split:
    - Train: Date <= train_end
    - Validation: train_end < Date <= val_end
    - Test Holdout: Date > val_end
    
    Verifies acceptance criteria:
    - No date overlap between sets
    - len(train) >= 30_000
    - len(test) >= 3_000
    
    Optionally saves indices to data/interim/*.parquet
    """
    if not np.issubdtype(df[DATE_COL].dtype, np.datetime64):
        df = df.copy()
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    dt_train_end = pd.to_datetime(train_end)
    dt_val_end = pd.to_datetime(val_end)

    train_mask = df[DATE_COL] <= dt_train_end
    val_mask = (df[DATE_COL] > dt_train_end) & (df[DATE_COL] <= dt_val_end)
    test_mask = df[DATE_COL] > dt_val_end

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Validations
    train_size = len(train_df)
    val_size = len(val_df)
    test_size = len(test_df)
    total_size = len(df)

    logger.info(
        "Temporal Split Sizes: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        train_size,
        train_size / total_size * 100,
        val_size,
        val_size / total_size * 100,
        test_size,
        test_size / total_size * 100,
    )

    if train_size < 30_000:
        raise ValueError(f"Train size check failed: {train_size} < 30,000 required rows.")
    if test_size < 3_000:
        raise ValueError(f"Test size check failed: {test_size} < 3,000 required rows.")

    # Date overlap verification
    train_max_date = train_df[DATE_COL].max()
    val_min_date = val_df[DATE_COL].min()
    val_max_date = val_df[DATE_COL].max()
    test_min_date = test_df[DATE_COL].min()

    if train_max_date >= val_min_date:
        raise ValueError(f"Train/Val date leakage! Train max ({train_max_date}) >= Val min ({val_min_date})")
    if val_max_date >= test_min_date:
        raise ValueError(f"Val/Test date leakage! Val max ({val_max_date}) >= Test min ({test_min_date})")

    logger.info("Date isolation verified: Train <= %s < Val <= %s < Test", train_max_date, val_max_date)

    if save_interim:
        save_path = interim_dir or DATA_INTERIM
        save_path.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({"index": train_df.index}).to_parquet(save_path / "train_idx.parquet")
        pd.DataFrame({"index": val_df.index}).to_parquet(save_path / "val_idx.parquet")
        pd.DataFrame({"index": test_df.index}).to_parquet(save_path / "test_idx.parquet")
        logger.info("Saved index parquets to %s", save_path)

    return train_df, val_df, test_df


def get_cv_folds(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    group_col: Optional[str] = None,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (train_idx, val_idx) fold splits for cross-validation within training set.
    Supports GroupKFold by spatial cluster or KFold.
    """
    if group_col and group_col in train_df.columns:
        # Impute missing groups for GroupKFold if any
        groups = train_df[group_col].fillna("unknown_group").astype(str).values
        gkf = GroupKFold(n_splits=n_splits)
        folds = list(gkf.split(train_df, groups=groups))
        logger.info("Constructed %d-fold GroupKFold on '%s'.", n_splits, group_col)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(kf.split(train_df))
        logger.info("Constructed %d-fold KFold (shuffle=True, seed=%d).", n_splits, random_state)
    return folds
