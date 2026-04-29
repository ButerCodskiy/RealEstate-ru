import numpy as np
import pandas as pd
from src.features.apartment import ApartmentFeatureExtractor


def test_apartment_feature_engineering():
    df = pd.DataFrame({
        "Общая_площадь": [50.0, 100.0],
        "Жилая_площадь": [30.0, 60.0],
        "Площадь_кухни": [10.0, 15.0],
        "Этаж": [1, 20],
        "Этажей_в_доме": [5, 20],
        "Высота_потолков": [2.6, 3.3],
        "Дата": ["2023-01-05", "2023-01-05"],
    })

    extractor = ApartmentFeatureExtractor()
    extractor.fit(df)
    res = extractor.transform(df)

    assert np.isclose(res["living_ratio"].iloc[0], 0.6)
    assert np.isclose(res["kitchen_ratio"].iloc[0], 0.2)
    assert res["is_first_floor"].iloc[0] == 1
    assert res["is_first_floor"].iloc[1] == 0
    assert res["is_top_floor"].iloc[0] == 0
    assert res["is_top_floor"].iloc[1] == 1
    assert res["is_penthouse"].iloc[1] == 1
    assert res["ceiling_category"].iloc[0] == "standard_le_2.75"
    assert res["ceiling_category"].iloc[1] == "high_gt_3.2"
    assert "log_total_area" in res.columns
    assert "months_since_min" in res.columns
