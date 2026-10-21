"""
predict.py
Command-line interface: enter a house's details, get a predicted price.

Usage:
    python predict.py          # interactive - it asks you questions
    python predict.py --demo   # runs three example houses, no typing needed
"""

import sys
import joblib
import pandas as pd

from features import add_features, ALL_FEATURES
from generate_data import LOCATION_MULTIPLIER

MODEL_PATH = "models/house_price_model.joblib"
LOCATIONS = list(LOCATION_MULTIPLIER.keys())


def predict_price(model, house):
    """house = dict of raw fields. Returns predicted price."""
    df = add_features(pd.DataFrame([house]))
    return float(model.predict(df[ALL_FEATURES])[0])


def ask_number(prompt, cast=float):
    while True:
        try:
            return cast(input(prompt))
        except ValueError:
            print("  Please enter a valid number.")


def ask_yes_no(prompt):
    while True:
        a = input(prompt + " (y/n): ").strip().lower()
        if a in ("y", "yes"):
            return 1
        if a in ("n", "no"):
            return 0
        print("  Please type y or n.")


def ask_location():
    print("Locations:")
    for i, loc in enumerate(LOCATIONS, 1):
        print(f"  {i}. {loc}")
    while True:
        c = ask_number("Choose location number: ", int)
        if 1 <= c <= len(LOCATIONS):
            return LOCATIONS[c - 1]
        print("  Please pick a number from the list.")


def interactive(model):
    print("\n=== House Price Predictor ===\n")
    house = {
        "size_sqft": ask_number("Size in square feet: "),
        "bedrooms": ask_number("Number of bedrooms: ", int),
        "bathrooms": ask_number("Number of bathrooms: ", int),
        "year_built": ask_number("Year built: ", int),
        "location": ask_location(),
        "has_garage": ask_yes_no("Has a garage?"),
        "has_garden": ask_yes_no("Has a garden?"),
        "distance_to_center_km": ask_number("Distance to city centre (km): "),
    }
    print(f"\nEstimated price: ${predict_price(model, house):,.0f}")


def demo(model):
    examples = [
        ("Large new house, Downtown", {
            "size_sqft": 3200, "bedrooms": 5, "bathrooms": 3, "year_built": 2021,
            "location": "Downtown", "has_garage": 1, "has_garden": 1,
            "distance_to_center_km": 2}),
        ("Average family home, Suburb South", {
            "size_sqft": 1700, "bedrooms": 3, "bathrooms": 2, "year_built": 1995,
            "location": "Suburb South", "has_garage": 1, "has_garden": 0,
            "distance_to_center_km": 8}),
        ("Small old house, Outskirts", {
            "size_sqft": 900, "bedrooms": 2, "bathrooms": 1, "year_built": 1960,
            "location": "Outskirts", "has_garage": 0, "has_garden": 0,
            "distance_to_center_km": 25}),
    ]
    print("\n=== Demo predictions ===")
    for label, house in examples:
        print(f"{label:38s} -> ${predict_price(model, house):>10,.0f}")


if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)
    if "--demo" in sys.argv:
        demo(model)
    else:
        interactive(model)
