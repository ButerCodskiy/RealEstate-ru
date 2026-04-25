"""
Configuration constants and parameters for ML Pipeline.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = DATA_RAW / "houseprices.csv"

# Target & Filtering thresholds
TARGET_COL = "Цена"
LOG_TARGET_COL = "log_Цена"
DATE_COL = "Дата"

PRICE_MIN = 500_000        # Minimum valid price (500k rub)
PRICE_MAX = 500_000_000    # Maximum valid price (500M rub)

# Temporal Split Dates
TRAIN_END_DATE = "2023-01-09"
VAL_END_DATE = "2023-01-11"

# All 44 expected raw columns
EXPECTED_RAW_COLUMNS = [
    "ID",
    "Ссылка",
    "Цена",
    "Дата",
    "Адрес",
    "Этаж",
    "Количество_комнат",
    "Балкон_или_лоджия",
    "Тип_комнат",
    "Общая_площадь",
    "Жилая_площадь",
    "Площадь_кухни",
    "Высота_потолков",
    "Санузел",
    "Окна",
    "Ремонт",
    "Мебель",
    "Тёплый_пол",
    "Отделка",
    "Техника",
    "Способ_продажи",
    "Вид_сделки",
    "Тип_дома",
    "В_доме",
    "Год_постройки",
    "Этажей_в_доме",
    "Пассажирский_лифт",
    "Грузовой_лифт",
    "Парковка",
    "Двор",
    "Название_новостройки",
    "Корпус_строение",
    "Официальный_застройщик",
    "Тип_участия",
    "Срок_сдачи",
    "Метро_1",
    "Метро_2",
    "Метро_3",
    "Расстояние_до_метро_1",
    "Расстояние_до_метро_2",
    "Расстояние_до_метро_3",
    "Широта",
    "Долгота",
    "Запланирован_снос",
]

# Columns to drop unconditionally (Group E)
COLUMNS_TO_DROP = [
    "ID",
    "Ссылка",
    "Адрес",
    "Тёплый_пол",        # 95.6% missing
    "Запланирован_снос", # 99.2% missing
]
