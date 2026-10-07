"""Build the lookup tables and prediction model used by the Flask app."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from predictor import (
    ASSETS,
    CAT_FEATURES,
    HGB_CONFIGS,
    MONTH_NUM,
    NUM_FEATURES,
    TARGET,
    build_season_table,
    fit_model,
    make_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = PROJECT_ROOT / "data" / "processed" / "master_train.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "regional_weather.csv"
PRICES_PATH = PROJECT_ROOT / "data" / "raw" / "market_prices.csv"

REGION_NAMES = {
    "amhara": "Amhara",
    "oromia": "Oromia",
    "snnpr": "SNNPR",
    "somali": "Somali",
    "tigray": "Tigray",
}
CROP_NAMES = {name: name for name in ("barley", "maize", "sorghum", "teff", "wheat")}
MONTH_NAME_TO_NUM = {name.lower(): number for name, number in MONTH_NUM.items()}


def standardize_categories(frame):
    frame = frame.copy()
    frame["region"] = frame["region"].astype(str).str.strip().str.lower().map(REGION_NAMES)
    if "crop_type" in frame:
        frame["crop_type"] = frame["crop_type"].astype(str).str.strip().str.lower().map(CROP_NAMES)
    return frame


def load_weather():
    weather = pd.read_csv(WEATHER_PATH)
    weather = standardize_categories(weather)
    weather["month"] = weather["month"].astype(str).str.strip().str.lower()
    weather["month_num"] = weather["month"].map(MONTH_NAME_TO_NUM)
    for column in ("year", "avg_temp_c", "monthly_rainfall_mm", "extreme_heat_days"):
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    weather = weather.dropna(subset=["region", "year", "month_num"])
    value_columns = ["avg_temp_c", "monthly_rainfall_mm", "extreme_heat_days"]
    weather[value_columns] = weather[value_columns].fillna(weather[value_columns].median())
    weather = (
        weather.drop_duplicates()
        .groupby(["region", "year", "month_num"], as_index=False)[value_columns]
        .mean()
    )
    weather["year"] = weather["year"].astype(int)
    weather["month_num"] = weather["month_num"].astype(int)
    weather["observed"] = 1
    return weather


def load_prices():
    prices = pd.read_csv(PRICES_PATH)
    prices = standardize_categories(prices)
    prices["year"] = pd.to_numeric(prices["year"], errors="coerce")
    prices["price_birr_per_quintal"] = pd.to_numeric(
        prices["price_birr_per_quintal"], errors="coerce"
    )
    prices = prices.dropna(subset=["region", "crop_type", "year"])
    prices["price_birr_per_quintal"] = prices["price_birr_per_quintal"].fillna(
        prices["price_birr_per_quintal"].median()
    )
    prices = (
        prices.drop_duplicates()
        .groupby(["crop_type", "region", "year"], as_index=False)["price_birr_per_quintal"]
        .mean()
    )
    prices["year"] = prices["year"].astype(int)
    return prices


def build_training_data(weather):
    source = pd.read_csv(TRAINING_PATH)
    source = standardize_categories(source)
    source["survey_year"] = pd.to_numeric(source["survey_year"], errors="coerce")
    source["planting_month_num"] = pd.to_numeric(source["planting_month_num"], errors="coerce")
    source = source.dropna(subset=["region", "crop_type", "survey_year", "planting_month_num", TARGET])
    source["survey_year"] = source["survey_year"].astype(int)
    source["planting_month_num"] = source["planting_month_num"].astype(int)

    season = build_season_table(weather)
    training = source.merge(
        season,
        on=["region", "survey_year", "planting_month_num"],
        how="left",
        validate="many_to_one",
    )
    training["is_meher_planting"] = (training["planting_month_num"] >= 6).astype(int)
    training["fert_x_improved_seed"] = (
        training["fertilizer_kg_per_ha"] * training["improved_seed_used"]
    )
    training["aridity_ratio"] = (
        training["season_mean_temp_c"] / (training["season_rain_total_mm"] + 1) * 100
    )
    features = CAT_FEATURES + NUM_FEATURES
    training = training[features + [TARGET]].replace([np.inf, -np.inf], np.nan)
    training[NUM_FEATURES] = training[NUM_FEATURES].apply(pd.to_numeric, errors="coerce")
    training[NUM_FEATURES] = training[NUM_FEATURES].fillna(training[NUM_FEATURES].median())
    training[TARGET] = pd.to_numeric(training[TARGET], errors="coerce")
    training = training.dropna(subset=[TARGET])
    if training.empty:
        raise ValueError(f"No usable training records were found in {TRAINING_PATH}.")
    return training


def main():
    required = (TRAINING_PATH, WEATHER_PATH, PRICES_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required source data is missing: " + ", ".join(missing))

    ASSETS.mkdir(parents=True, exist_ok=True)
    weather = load_weather()
    prices = load_prices()
    training = build_training_data(weather)
    training.to_csv(ASSETS / "training_data.csv", index=False)
    weather.to_csv(ASSETS / "weather_clean.csv", index=False)
    prices.to_csv(ASSETS / "prices_clean.csv", index=False)

    config_name = "tuned"
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        make_pipeline(HGB_CONFIGS[config_name]),
        training[CAT_FEATURES + NUM_FEATURES],
        training[TARGET],
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
    )
    model = fit_model(training, config_name)
    joblib.dump(model, ASSETS / "model.joblib")
    card = {
        "chosen_config": config_name,
        "cv_rmse_mean": float(-scores.mean()),
        "cv_rmse_sd": float(scores.std()),
        "training_rows": int(len(training)),
        "target": TARGET,
        "features": CAT_FEATURES + NUM_FEATURES,
    }
    (ASSETS / "model_card.json").write_text(
        json.dumps(card, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Built app assets from {len(training)} training rows in {ASSETS}.")


if __name__ == "__main__":
    main()
