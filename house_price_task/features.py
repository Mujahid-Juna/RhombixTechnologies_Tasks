"""
features.py
Feature engineering shared by the training script and the prediction CLI,
so a new house goes through exactly the same steps as the training data.
"""

import numpy as np
import pandas as pd

CURRENT_YEAR = 2025

NUMERIC_FEATURES = [
    "size_sqft", "bedrooms", "bathrooms", "house_age",
    "distance_to_center_km", "has_garage", "has_garden",
    "total_rooms", "size_per_room", "is_new",
]
CATEGORICAL_FEATURES = ["location"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"


def add_features(df):
    """Create new, more informative columns from the raw ones."""
    df = df.copy()
    df["house_age"] = CURRENT_YEAR - df["year_built"]
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["size_per_room"] = df["size_sqft"] / df["total_rooms"].replace(0, np.nan)
    df["is_new"] = (df["house_age"] <= 5).astype(float)
    # keep NaN where house_age is unknown (is_new would wrongly say 0)
    df.loc[df["house_age"].isna(), "is_new"] = np.nan
    return df
