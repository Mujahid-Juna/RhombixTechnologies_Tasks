"""
generate_data.py
Generates a realistic synthetic housing dataset.

Why synthetic? Real housing datasets (Kaggle, California Housing) need an
internet download, and the classic Boston Housing dataset has been removed
from scikit-learn. Generating data locally keeps this project runnable
anywhere with zero downloads.

The prices follow real-estate logic (bigger = pricier, newer = pricier,
better neighbourhood = pricier, far from centre = cheaper) plus random noise,
and a few missing values are injected so the data-cleaning step is real.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=11)
N_SAMPLES = 2000

# Price multiplier for each neighbourhood (location effect)
LOCATION_MULTIPLIER = {
    "Downtown": 1.35,
    "Suburb North": 1.10,
    "Suburb South": 1.00,
    "Riverside": 1.20,
    "Outskirts": 0.80,
}


def generate_dataset(n=N_SAMPLES):
    location = RNG.choice(
        list(LOCATION_MULTIPLIER.keys()), size=n, p=[0.15, 0.25, 0.25, 0.15, 0.20]
    )
    size_sqft = RNG.normal(1700, 600, n).clip(500, 5000)
    bedrooms = np.clip(np.round(size_sqft / 600 + RNG.normal(0, 0.7, n)), 1, 7).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.6 + RNG.normal(0, 0.5, n)), 1, 5).astype(int)
    year_built = RNG.integers(1950, 2025, n)
    has_garage = RNG.choice([0, 1], size=n, p=[0.3, 0.7])
    has_garden = RNG.choice([0, 1], size=n, p=[0.5, 0.5])
    distance_to_center_km = RNG.exponential(8, n).clip(0.5, 40)

    age = 2025 - year_built
    loc_mult = np.array([LOCATION_MULTIPLIER[x] for x in location])

    price = (
        60000
        + 135 * size_sqft
        + 6000 * bedrooms
        + 9000 * bathrooms
        - 700 * age
        + 12000 * has_garage
        + 8000 * has_garden
        - 1500 * distance_to_center_km
    ) * loc_mult
    price = price * RNG.normal(1.0, 0.07, n)  # market noise
    price = price.clip(40000, None).round(-2)

    df = pd.DataFrame({
        "size_sqft": size_sqft.round(0),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "location": location,
        "has_garage": has_garage,
        "has_garden": has_garden,
        "distance_to_center_km": distance_to_center_km.round(2),
        "price": price,
    })

    # Inject realistic missing values (listings often have gaps)
    for col, frac in [("size_sqft", 0.03), ("bathrooms", 0.04),
                      ("distance_to_center_km", 0.03), ("year_built", 0.02)]:
        idx = RNG.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("house_data.csv", index=False)
    print(f"Generated {len(df)} houses -> house_data.csv")
    print(df.describe().round(1))
    print("\nMissing values:\n", df.isnull().sum())
