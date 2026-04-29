import numpy as np
import pandas as pd
from src.features.geo import haversine_distance, GeoFeatureExtractor
from src.config import TARGET_COL


def test_haversine_known_distance():
    # Distance between Moscow Kremlin (55.7520, 37.6175) and Khamovniki (55.7300, 37.5800) is approx 3.4 km
    d = haversine_distance(55.7520, 37.6175, 55.7300, 37.5800)
    assert 3.0 < d < 4.0
    # Zero distance to self
    d_self = haversine_distance(55.75, 37.61, 55.75, 37.61)
    assert np.isclose(d_self, 0.0)


def test_geo_feature_extractor_fit_transform():
    df_toy = pd.DataFrame({
        "Широта": [55.75, 55.76, 55.74, 55.70, 55.80],
        "Долгота": [37.61, 37.62, 37.60, 37.50, 37.70],
        TARGET_COL: [10_000_000.0, 15_000_000.0, 20_000_000.0, 12_000_000.0, 14_000_000.0]
    })

    extractor = GeoFeatureExtractor(n_clusters=2)
    extractor.fit(df_toy)
    df_out = extractor.transform(df_toy)

    assert "dist_to_center_km" in df_out.columns
    assert "log_dist_to_center" in df_out.columns
    assert "geo_cluster" in df_out.columns
    assert (df_out["dist_to_center_km"] >= 0).all()
    assert (df_out["geo_cluster"].nunique() <= 2)
