"""
Geographical feature extraction module (H-GEO).
Calculates Haversine distance to city center, log-distance, and spatial clusters via KMeans.
Strictly fits center coordinates and KMeans model on training data only.
"""
from typing import Tuple, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import TARGET_COL

logger = logging.getLogger(__name__)


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """
    Computes great-circle distance in kilometers between coordinates using Haversine formula.
    """
    r = 6371.0  # Earth radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return r * c


class GeoFeatureExtractor:
    """
    Extracts spatial and location-based features:
    - dist_to_center_km
    - log_dist_to_center
    - geo_cluster (KMeans spatial clustering)
    """

    def __init__(self, n_clusters: int = 30, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.center_lat = None
        self.center_lon = None
        self.kmeans = None

    def fit(self, train_df: pd.DataFrame) -> "GeoFeatureExtractor":
        logger.info("Fitting GeoFeatureExtractor on %d training samples...", len(train_df))

        # 1. Compute empirical city center: median lat/lon of top 10% most expensive properties
        top10_threshold = train_df[TARGET_COL].quantile(0.90)
        top10_sub = train_df[train_df[TARGET_COL] >= top10_threshold]

        self.center_lat = float(top10_sub["Широта"].median())
        self.center_lon = float(top10_sub["Долгота"].median())
        logger.info("Computed city center: lat=%.6f, lon=%.6f (cutoff: %.0f RUB)", self.center_lat, self.center_lon, top10_threshold)

        # 2. Fit spatial KMeans clustering
        coords = train_df[["Широта", "Долгота"]].values
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
        self.kmeans.fit(coords)
        logger.info("Fitted spatial KMeans with %d clusters.", self.n_clusters)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.center_lat is None or self.kmeans is None:
            raise RuntimeError("GeoFeatureExtractor must be fitted on training data before calling transform!")

        out = df.copy()
        lats = out["Широта"].values
        lons = out["Долгота"].values

        # 1. Distances to center
        dist_km = haversine_distance(lats, lons, self.center_lat, self.center_lon)
        out["dist_to_center_km"] = dist_km
        out["log_dist_to_center"] = np.log1p(dist_km)

        # 2. Spatial KMeans clusters
        coords = out[["Широта", "Долгота"]].values
        out["geo_cluster"] = self.kmeans.predict(coords)

        return out
