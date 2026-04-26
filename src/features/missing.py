"""
Missing value handling strategies (Groups A, B, C, D, E) per PLAN.md.
Strictly avoids data leakage: all imputation statistics are fitted on the train split only.
"""
from typing import Tuple, Optional, Dict
import logging
import re
import numpy as np
import pandas as pd

from src.config import COLUMNS_TO_DROP, DATE_COL

logger = logging.getLogger(__name__)


def parse_ceiling_height(val) -> float:
    """Parses ceiling height string (e.g., '2.7м', '264м') into numeric meters."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace("м", "").replace(",", ".").strip()
    try:
        f = float(s)
        if f > 50.0:  # e.g. 264 -> 2.64m
            f = f / 100.0
        if f < 1.8 or f > 6.0:  # Filter out anomalies like 0m or extreme outliers
            return np.nan
        return f
    except (ValueError, TypeError):
        return np.nan


def parse_lift_count(val) -> float:
    """Parses passenger/freight lift count (e.g. '1', '2', 'нет') into float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ["нет", "0", "0.0"]:
        return 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def parse_months_to_delivery(row) -> float:
    """Computes months until delivery relative to listing Date."""
    val = row.get("Срок_сдачи")
    if pd.isna(val):
        return 0.0
    s = str(val).strip().lower()
    if "сдан" in s:
        return 0.0
    m = re.search(r"(\d)\s*кв\.\s*(\d{4})", s)
    if m:
        quarter = int(m.group(1))
        year = int(m.group(2))
        month = quarter * 3
        current_dt = row[DATE_COL]
        diff_months = (year - current_dt.year) * 12 + (month - current_dt.month)
        return max(0.0, float(diff_months))
    return 0.0


class MissingValueImputer:
    """
    Stateful imputer for missing values.
    Fitted exclusively on train data; transforms train, val, and test data identically.
    """

    def __init__(self, n_area_quantiles: int = 10):
        self.n_area_quantiles = n_area_quantiles
        self.area_bins = None
        self.living_medians_3way = {}
        self.living_medians_2way = {}
        self.global_living_median = None

        self.kitchen_medians_2way = {}
        self.global_kitchen_median = None

        self.ceiling_medians = {}
        self.global_ceiling_median = None

        self.year_built_medians = {}
        self.global_year_built_median = None

    def fit(self, train_df: pd.DataFrame) -> "MissingValueImputer":
        logger.info("Fitting MissingValueImputer on training set (%d rows)...", len(train_df))
        df = train_df.copy()

        # 1. Total area quantiles
        _, self.area_bins = pd.qcut(
            df["Общая_площадь"],
            q=self.n_area_quantiles,
            retbins=True,
            duplicates="drop",
        )
        # Ensure bin boundaries cover extremes
        self.area_bins[0] = -np.inf
        self.area_bins[-1] = np.inf

        area_bin_series = pd.cut(df["Общая_площадь"], bins=self.area_bins, labels=False)
        rooms_series = df["Количество_комнат"].fillna("unknown").astype(str)
        house_type_series = df["Тип_дома"].fillna("unknown").astype(str)

        # 2. Living area medians
        self.global_living_median = float(df["Жилая_площадь"].median())
        df_living = df[["Жилая_площадь"]].copy()
        df_living["bin"] = area_bin_series
        df_living["rooms"] = rooms_series
        df_living["house_type"] = house_type_series

        g3 = df_living.groupby(["bin", "rooms", "house_type"])["Жилая_площадь"].median()
        self.living_medians_3way = g3.to_dict()

        g2 = df_living.groupby(["bin", "rooms"])["Жилая_площадь"].median()
        self.living_medians_2way = g2.to_dict()

        # 3. Kitchen area medians
        self.global_kitchen_median = float(df["Площадь_кухни"].median())
        df_kitchen = df[["Площадь_кухни"]].copy()
        df_kitchen["bin"] = area_bin_series
        df_kitchen["rooms"] = rooms_series

        gk = df_kitchen.groupby(["bin", "rooms"])["Площадь_кухни"].median()
        self.kitchen_medians_2way = gk.to_dict()

        # 4. Ceiling height medians
        parsed_ceilings = df["Высота_потолков"].apply(parse_ceiling_height)
        self.global_ceiling_median = float(parsed_ceilings.median())
        df_ceil = pd.DataFrame({"ceiling": parsed_ceilings, "house_type": house_type_series})
        self.ceiling_medians = df_ceil.groupby("house_type")["ceiling"].median().to_dict()

        # 5. Year built medians
        self.global_year_built_median = float(df["Год_постройки"].median())
        df_year = pd.DataFrame({"year": df["Год_постройки"], "house_type": house_type_series})
        self.year_built_medians = df_year.groupby("house_type")["year"].median().to_dict()

        logger.info("MissingValueImputer fitted successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        out = df.copy()

        # --- ГРУППА A: Флаг + Заполнение ---
        # Балкон или лоджия
        out["balcony_missing"] = out["Балкон_или_лоджия"].isna().astype(int)
        out["Балкон_или_лоджия"] = out["Балкон_или_лоджия"].fillna("Не указан")

        # Ремонт
        out["repair_missing"] = out["Ремонт"].isna().astype(int)
        out["Ремонт"] = out["Ремонт"].fillna("Не указан")

        # Парковка
        out["parking_missing"] = out["Парковка"].isna().astype(int)
        out["Парковка"] = out["Парковка"].fillna("Не указана")

        # Двор
        out["yard_missing"] = out["Двор"].isna().astype(int)
        out["Двор"] = out["Двор"].fillna("Не указан")

        # Мебель & Техника
        out["is_furnished_mentioned"] = out["Мебель"].notna().astype(int)
        out["Мебель"] = out["Мебель"].fillna("Не указано")

        out["has_appliances_mentioned"] = out["Техника"].notna().astype(int)
        out["Техника"] = out["Техника"].fillna("Не указано")

        # Отделка
        out["otdelka_missing"] = out["Отделка"].isna().astype(int)
        out["Отделка"] = out["Отделка"].fillna("Не указана")

        # Высота потолков
        parsed_ceil = out["Высота_потолков"].apply(parse_ceiling_height)
        out["ceiling_missing"] = parsed_ceil.isna().astype(int)
        ht_series = out["Тип_дома"].fillna("unknown").astype(str)
        imputed_ceil = ht_series.map(self.ceiling_medians).fillna(self.global_ceiling_median)
        out["Высота_потолков"] = parsed_ceil.fillna(imputed_ceil)

        # Год постройки
        out["year_built_missing"] = out["Год_постройки"].isna().astype(int)
        imputed_year = ht_series.map(self.year_built_medians).fillna(self.global_year_built_median)
        out["Год_постройки"] = out["Год_постройки"].fillna(imputed_year)

        # Жилая площадь и площадь кухни
        area_bins = pd.cut(out["Общая_площадь"], bins=self.area_bins, labels=False)
        rooms = out["Количество_комнат"].fillna("unknown").astype(str)

        # Жилая площадь
        out["living_area_missing"] = out["Жилая_площадь"].isna().astype(int)
        living_imp_3way = [
            self.living_medians_3way.get((b, r, h), np.nan)
            for b, r, h in zip(area_bins, rooms, ht_series)
        ]
        living_imp_2way = [
            self.living_medians_2way.get((b, r), np.nan)
            for b, r in zip(area_bins, rooms)
        ]
        living_filler = pd.Series(living_imp_3way, index=out.index).fillna(
            pd.Series(living_imp_2way, index=out.index)
        ).fillna(self.global_living_median)
        out["Жилая_площадь"] = out["Жилая_площадь"].fillna(living_filler)

        # Площадь кухни
        out["kitchen_missing"] = out["Площадь_кухни"].isna().astype(int)
        kitchen_imp_2way = [
            self.kitchen_medians_2way.get((b, r), np.nan)
            for b, r in zip(area_bins, rooms)
        ]
        kitchen_filler = pd.Series(kitchen_imp_2way, index=out.index).fillna(
            self.global_kitchen_median
        )
        out["Площадь_кухни"] = out["Площадь_кухни"].fillna(kitchen_filler)

        # --- ГРУППА B: Заполнение нулём ---
        parsed_p_lift = out["Пассажирский_лифт"].apply(parse_lift_count).fillna(0.0)
        # Если этажей <= 5, лифта обычно нет
        low_rise_mask = out["Этажей_в_доме"].fillna(0) <= 5
        parsed_p_lift[low_rise_mask] = parsed_p_lift[low_rise_mask].fillna(0.0)
        out["Пассажирский_лифт"] = parsed_p_lift

        out["Грузовой_лифт"] = out["Грузовой_лифт"].apply(parse_lift_count).fillna(0.0)

        # --- ГРУППА C: Новостройки ---
        out["is_new_building"] = out["Название_новостройки"].notna().astype(int)
        out["Официальный_застройщик"] = out["Официальный_застройщик"].fillna("secondary_market")
        out["Тип_участия"] = out["Тип_участия"].fillna("Не указан")
        out["Корпус_строение"] = out["Корпус_строение"].fillna("Не указан")
        out["months_to_delivery"] = out.apply(parse_months_to_delivery, axis=1)

        # --- ГРУППА D: Нативные NaNs сохраняются ---
        # (Санузел, Окна, Способ_продажи, Вид_сделки, В_доме, Тип_комнат, Метро_*, Расстояние_до_метро_*)
        # Оставляем как есть для бустингов

        # --- ГРУППА E: Удаление колонок ---
        drop_cols_existing = [c for c in COLUMNS_TO_DROP if c in out.columns]
        out = out.drop(columns=drop_cols_existing)

        # Проверка целостности
        if len(out) != initial_len:
            raise RuntimeError(
                f"Integrity error! Row count changed during missing value processing: {initial_len} -> {len(out)}"
            )

        return out
