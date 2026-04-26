"""
Apartment and temporal feature engineering module (H-APT & H-TIME).
Generates:
- living_ratio, kitchen_ratio, non_living_sqm
- floor_ratio, is_first_floor, is_top_floor, is_penthouse
- log_total_area, log_kitchen_area
- ceiling_category
- year, quarter, month, day_of_week, months_since_min
"""
from typing import Optional
import logging
import numpy as np
import pandas as pd

from src.config import DATE_COL

logger = logging.getLogger(__name__)


class ApartmentFeatureExtractor:
    """
    Extracts apartment-level geometric ratios and temporal indicators.
    Fitted on training set to lock temporal references.
    """

    def __init__(self):
        self.min_train_date = None

    def fit(self, train_df: pd.DataFrame) -> "ApartmentFeatureExtractor":
        dts = pd.to_datetime(train_df[DATE_COL])
        self.min_train_date = dts.min()
        logger.info("ApartmentFeatureExtractor fitted (reference min date: %s).", self.min_train_date.date())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # 1. Area ratios (H-APT)
        tot_area = np.maximum(out["Общая_площадь"].values, 1.0)
        liv_area = np.maximum(out["Жилая_площадь"].values, 0.0)
        kitch_area = np.maximum(out["Площадь_кухни"].values, 0.0)

        out["living_ratio"] = np.clip(liv_area / tot_area, 0.0, 1.0)
        out["kitchen_ratio"] = np.clip(kitch_area / tot_area, 0.0, 1.0)
        out["non_living_sqm"] = np.maximum(0.0, tot_area - liv_area)

        # 2. Floor metrics
        floor = out["Этаж"].values
        total_floors = np.maximum(out["Этажей_в_доме"].fillna(1).values, 1.0)

        out["floor_ratio"] = np.clip(floor / total_floors, 0.0, 1.0)
        out["is_first_floor"] = (floor == 1).astype(int)
        out["is_top_floor"] = (floor == total_floors).astype(int)
        out["is_penthouse"] = ((out["floor_ratio"] >= 0.95) & (floor > 10)).astype(int)

        # 3. Logarithmic transformations
        out["log_total_area"] = np.log1p(tot_area)
        out["log_kitchen_area"] = np.log1p(kitch_area)

        # 4. Ceiling category
        ceil_vals = out["Высота_потолков"].values
        ceil_cat = np.full(len(out), "comfort_2.8-3.2", dtype=object)
        ceil_cat[ceil_vals <= 2.75] = "standard_le_2.75"
        ceil_cat[ceil_vals > 3.2] = "high_gt_3.2"
        out["ceiling_category"] = ceil_cat

        # 5. Temporal features (H-TIME)
        dts = pd.to_datetime(out[DATE_COL])
        out["year"] = dts.dt.year
        out["quarter"] = dts.dt.quarter
        out["month"] = dts.dt.month
        out["day_of_week"] = dts.dt.dayofweek

        ref_date = self.min_train_date or dts.min()
        out["months_since_min"] = (dts.dt.year - ref_date.year) * 12 + (dts.dt.month - ref_date.month)

        return out
