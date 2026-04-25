"""
Data loader and schema validation module.
"""
from pathlib import Path
from typing import Optional, Tuple
import logging
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from src.config import (
    RAW_DATA_PATH,
    EXPECTED_RAW_COLUMNS,
    TARGET_COL,
    LOG_TARGET_COL,
    DATE_COL,
    PRICE_MIN,
    PRICE_MAX,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_raw_schema(df: pd.DataFrame) -> None:
    """
    Validates that the input DataFrame contains all 44 expected columns.
    Raises ValueError if any required column is missing.
    """
    missing_cols = set(EXPECTED_RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Schema validation failed! Missing {len(missing_cols)} expected columns: {sorted(missing_cols)}"
        )
    extra_cols = set(df.columns) - set(EXPECTED_RAW_COLUMNS)
    if extra_cols:
        logger.warning("Found %d unexpected columns in input data: %s", len(extra_cols), sorted(extra_cols))
    logger.info("Schema validation passed: all %d expected columns present.", len(EXPECTED_RAW_COLUMNS))


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads raw CSV data and verifies schema integrity.
    """
    path = filepath or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    logger.info("Loading raw dataset from %s ...", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns.", len(df), len(df.columns))

    validate_raw_schema(df)
    return df


def filter_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Filters out invalid target records:
    - Min price: PRICE_MIN (500 000 rub)
    - Max price: PRICE_MAX (500 000 000 rub)
    - Applies log1p transformation to TARGET_COL
    - Converts DATE_COL to datetime64[ns]
    
    Returns (filtered_df, filter_stats)
    """
    total_raw = len(df)
    raw_min_price = float(df[TARGET_COL].min())
    raw_max_price = float(df[TARGET_COL].max())
    raw_skew = float(skew(df[TARGET_COL].dropna()))
    raw_kurt = float(kurtosis(df[TARGET_COL].dropna()))

    # Apply filtering condition
    valid_mask = (df[TARGET_COL] >= PRICE_MIN) & (df[TARGET_COL] <= PRICE_MAX)
    filtered_df = df[valid_mask].copy().reset_index(drop=True)
    dropped_rows = total_raw - len(filtered_df)

    # Log target
    filtered_df[LOG_TARGET_COL] = np.log1p(filtered_df[TARGET_COL])
    filtered_df[DATE_COL] = pd.to_datetime(filtered_df[DATE_COL])

    log_skew = float(skew(filtered_df[LOG_TARGET_COL]))
    log_kurt = float(kurtosis(filtered_df[LOG_TARGET_COL]))

    stats = {
        "total_raw": total_raw,
        "clean_rows": len(filtered_df),
        "dropped_rows": dropped_rows,
        "dropped_pct": dropped_rows / total_raw * 100,
        "raw_min_price": raw_min_price,
        "raw_max_price": raw_max_price,
        "raw_skew": raw_skew,
        "raw_kurt": raw_kurt,
        "clean_min_price": float(filtered_df[TARGET_COL].min()),
        "clean_max_price": float(filtered_df[TARGET_COL].max()),
        "clean_mean_price": float(filtered_df[TARGET_COL].mean()),
        "clean_median_price": float(filtered_df[TARGET_COL].median()),
        "log_skew": log_skew,
        "log_kurt": log_kurt,
    }

    logger.info(
        "Filtered dataset: %d rows retained (%d dropped, %.2f%%).",
        len(filtered_df),
        dropped_rows,
        stats["dropped_pct"],
    )
    logger.info("log1p(Price) skewness: %.3f (target: < 1.5), kurtosis: %.3f", log_skew, log_kurt)
    return filtered_df, stats
