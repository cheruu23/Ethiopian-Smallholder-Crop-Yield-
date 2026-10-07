"""Core logic of the crop-yield demo: validation, automatic weather/price lookup, prediction, charts.

The web layer (app.py) only calls Predictor.run(form). Everything here can be tested without a browser.
"""
import base64
import io
import json
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ASSETS = Path(__file__).resolve().parent / "assets"

REGIONS = ["Amhara", "Oromia", "SNNPR", "Somali", "Tigray"]
CROPS = ["barley", "maize", "sorghum", "teff", "wheat"]
YEARS = [2021, 2022, 2023, 2024]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}
PLANTING_MONTHS = ["Feb", "Mar", "Jun", "Jul", "Aug"]       # the months that exist in the training data
SEASON_LENGTH = 4                                           # growing season = planting month + next 3 months

TARGET = "yield_tons_per_ha"
CAT_FEATURES = ["region", "crop_type"]
# The model uses only what the user types plus what the app looks up. Plot-reported rainfall is NOT used
# (the form has no rainfall box, and notebook 02 / B4.3 showed it carries almost no signal).
NUM_FEATURES = ["survey_year", "altitude_m", "farm_size_ha", "fertilizer_kg_per_ha", "improved_seed_used", "pest_disease_flag",
                "soil_quality_index", "labor_days_per_ha", "distance_to_market_km",
                "season_mean_temp_c", "season_rain_total_mm", "season_extreme_heat_days", "season_temp_dev_c",
                "planting_month_num", "is_meher_planting", "fert_x_improved_seed", "aridity_ratio"]

# HistGradientBoosting settings: the notebook's first model, and the settings its random search chose
HGB_CONFIGS = {
    "original": dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31),
    "tuned": dict(min_samples_leaf=10, max_leaf_nodes=15, max_iter=300, learning_rate=0.1, l2_regularization=1.0),
}

# user-typed numeric fields: label, unit, hard minimum, hard maximum, whether 0 is allowed
FIELDS = {
    "altitude_m": ("Altitude", "m", 0, 5000, True),
    "farm_size_ha": ("Farm size", "ha", 0, 1000, False),
    "fertilizer_kg_per_ha": ("Fertilizer", "kg per ha", 0, 2000, True),
    "soil_quality_index": ("Soil quality", "0 to 1", 0, 1, True),
    "labor_days_per_ha": ("Labour", "days per ha", 0, 1000, True),
    "distance_to_market_km": ("Distance to market", "km", 0, 1000, True),
}
DEFAULTS = {"region": "Amhara", "crop_type": "maize", "survey_year": "2023", "planting_month": "Jun", "altitude_m": "1800",
            "farm_size_ha": "1.2", "fertilizer_kg_per_ha": "40", "improved_seed_used": "1", "pest_disease_flag": "0",
            "soil_quality_index": "0.6", "labor_days_per_ha": "60", "distance_to_market_km": "8"}

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00", "grey": "#999999"}


# ----------------------------------------------------------------------------- model definition
def make_pipeline(config, num_features=None):
    num_features = num_features or NUM_FEATURES
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES), ("num", "passthrough", num_features)])
    return Pipeline([("pre", pre), ("model", HistGradientBoostingRegressor(random_state=42, **config))])


def fit_model(training_df, config_name="tuned"):
    pipe = make_pipeline(HGB_CONFIGS[config_name])
    pipe.fit(training_df[CAT_FEATURES + NUM_FEATURES], training_df[TARGET])
    return pipe


# ----------------------------------------------------------------------------- weather: season features
def build_season_table(weather):
    """One row per region + year + planting month: stats over planting month and the next 3 months (same rule as notebook 01)."""
    rows = []
    last_start = 12 - SEASON_LENGTH + 1
    for (region, year), g in weather.groupby(["region", "year"]):
        g = g.set_index("month_num")
        for start in range(1, last_start + 1):
            w = g.reindex(range(start, start + SEASON_LENGTH))
            rows.append({"region": region, "survey_year": int(year), "planting_month_num": start,
                         "season_mean_temp_c": w.avg_temp_c.mean(), "season_rain_total_mm": w.monthly_rainfall_mm.sum(),
                         "season_extreme_heat_days": int(w.extreme_heat_days.sum()), "weather_months_observed": int(w.observed.sum())})
    season = pd.DataFrame(rows)
    typical = season.groupby(["region", "planting_month_num"])["season_mean_temp_c"].transform("mean")
    season["season_temp_dev_c"] = season["season_mean_temp_c"] - typical
    return season


class Predictor:
    def __init__(self):
        self.weather = pd.read_csv(ASSETS / "weather_clean.csv")
        self.prices = pd.read_csv(ASSETS / "prices_clean.csv")
        self.training = pd.read_csv(ASSETS / "training_data.csv")
        self.card = json.load(open(ASSETS / "model_card.json"))
        self.season = build_season_table(self.weather).set_index(["region", "survey_year", "planting_month_num"])
        self.price_index = self.prices.set_index(["crop_type", "region", "year"])["price_birr_per_quintal"]
        self.model = self._load_model()
        self.ranges = {c: (float(self.training[c].min()), float(self.training[c].max())) for c in FIELDS}
        self.avg_region_crop = self.training.groupby(["region", "crop_type"])[TARGET].mean()
        self.avg_crop = self.training.groupby("crop_type")[TARGET].mean()
        self.rmse = float(self.card["cv_rmse_mean"])

    def _load_model(self):
        try:
            return joblib.load(ASSETS / "model.joblib")
        except Exception as exc:       # e.g. different scikit-learn version: just retrain (takes a few seconds)
            print("Could not load model.joblib (", exc, ") - retraining from assets/training_data.csv")
            return fit_model(self.training, self.card["chosen_config"])

    # ------------------------------------------------------------------------- 1. validation
    def validate(self, form):
        """Returns (clean_values, errors, warnings). Never raises."""
        errors, warnings, clean = [], [], {}
        get = lambda k: str(form.get(k, "") if form.get(k, "") is not None else "").strip()

        for key, options, label in [("region", REGIONS, "region"), ("crop_type", CROPS, "crop type"), ("planting_month", PLANTING_MONTHS, "planting month")]:
            value = get(key)
            if value == "":
                errors.append(f"Please choose a {label}.")
            elif value not in options:
                errors.append(f"'{value}' is not a valid {label}. Please pick one from the list.")
            else:
                clean[key] = value
        year = get("survey_year")
        if year == "":
            errors.append("Please choose a survey year.")
        elif not year.isdigit() or int(year) not in YEARS:
            errors.append(f"Survey year must be one of {YEARS[0]}-{YEARS[-1]}.")
        else:
            clean["survey_year"] = int(year)
        for key, label in [("improved_seed_used", "improved seed"), ("pest_disease_flag", "pest/disease")]:
            value = get(key)
            if value not in ("0", "1"):
                errors.append(f"Please answer Yes or No for {label}.")
            else:
                clean[key] = int(value)

        for key, (label, unit, lo, hi, zero_ok) in FIELDS.items():
            text = get(key)
            if text == "":
                errors.append(f"{label} is empty - please enter a number ({unit}).")
                continue
            try:
                value = float(text)
            except ValueError:
                errors.append(f"{label} must be a number (you typed '{text}').")
                continue
            if math.isnan(value) or math.isinf(value):
                errors.append(f"{label} must be a real number.")
            elif value < lo or (value == 0 and not zero_ok):
                errors.append(f"{label} must be greater than 0 ({unit})." if not zero_ok else f"{label} cannot be negative ({unit}).")
            elif value > hi:
                errors.append(f"{label} of {value:g} {unit} is not realistic (maximum accepted is {hi:g}).")
            else:
                clean[key] = value
                tmin, tmax = self.ranges[key]
                if value > tmax * 1.001 or value < tmin * 0.999:
                    warnings.append(f"{label} ({value:g} {unit}) is outside the range the model learned from ({tmin:g} to {tmax:g}); "
                                    f"the prediction for this plot is less reliable.")
        return clean, errors, warnings

    # ------------------------------------------------------------------------- 2. automatic lookup
    def lookup(self, v):
        """Weather + price for the chosen region / crop / year / planting month. The user never types these."""
        start = MONTH_NUM[v["planting_month"]]
        s = self.season.loc[(v["region"], v["survey_year"], start)]
        months = [MONTHS[start - 1 + i] for i in range(SEASON_LENGTH)]
        price = float(self.price_index.loc[(v["crop_type"], v["region"], v["survey_year"])])
        info = {"window": f"{months[0]}-{months[-1]}", "temp": float(s.season_mean_temp_c), "rain": float(s.season_rain_total_mm),
                "heat": int(s.season_extreme_heat_days), "dev": float(s.season_temp_dev_c), "observed": int(s.weather_months_observed), "price": price}
        return s, info

    # ------------------------------------------------------------------------- 3. prediction
    def make_rows(self, v, season_row, fertilizer=None, seed=None):
        fert = v["fertilizer_kg_per_ha"] if fertilizer is None else fertilizer
        seed = v["improved_seed_used"] if seed is None else seed
        n = len(np.atleast_1d(fert))
        start = MONTH_NUM[v["planting_month"]]
        rows = pd.DataFrame({
            "region": v["region"], "crop_type": v["crop_type"], "survey_year": v["survey_year"], "altitude_m": v["altitude_m"],
            "farm_size_ha": v["farm_size_ha"], "fertilizer_kg_per_ha": np.atleast_1d(fert) * 1.0, "improved_seed_used": seed,
            "pest_disease_flag": v["pest_disease_flag"], "soil_quality_index": v["soil_quality_index"],
            "labor_days_per_ha": v["labor_days_per_ha"], "distance_to_market_km": v["distance_to_market_km"],
            "season_mean_temp_c": season_row.season_mean_temp_c, "season_rain_total_mm": season_row.season_rain_total_mm,
            "season_extreme_heat_days": season_row.season_extreme_heat_days, "season_temp_dev_c": season_row.season_temp_dev_c,
            "planting_month_num": start, "is_meher_planting": int(start >= 6)}, index=range(n))
        rows["fert_x_improved_seed"] = rows["fertilizer_kg_per_ha"] * rows["improved_seed_used"]
        rows["aridity_ratio"] = rows["season_mean_temp_c"] / (rows["season_rain_total_mm"] + 1) * 100
        return rows[CAT_FEATURES + NUM_FEATURES]

    def predict_yield(self, rows):
        return np.clip(self.model.predict(rows), 0, None)

    # ------------------------------------------------------------------------- 4. charts
    @staticmethod
    def _png(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def chart_compare(self, v, pred):
        labels = ["Your plot\n(predicted)", f"Average {v['crop_type']}\nin {v['region']}", f"Average {v['crop_type']}\nall regions"]
        values = [pred, float(self.avg_region_crop.loc[(v["region"], v["crop_type"])]), float(self.avg_crop.loc[v["crop_type"]])]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=[OI["blue"], OI["orange"], OI["grey"]], width=0.6)
        for b, val in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, val + max(values) * 0.015, f"{val:.2f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.18)
        ax.set_ylabel("Yield (tons per hectare)", fontsize=12)
        ax.set_title("Your prediction vs the typical plot", fontsize=13, fontweight="bold", loc="left")
        ax.tick_params(labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        return self._png(fig)

    def chart_whatif(self, v, season_row, pred):
        # stop at 120 kg/ha: above that the training data is thin (and includes capped outliers), so the curve would not be trustworthy
        cap = min(self.ranges["fertilizer_kg_per_ha"][1], 120)
        grid = np.linspace(0, cap, 40)
        fig, ax = plt.subplots(figsize=(6, 4))
        for seed, color, name in [(1, OI["green"], "With improved seed"), (0, OI["vermillion"], "Without improved seed")]:
            y = self.predict_yield(self.make_rows(v, season_row, fertilizer=grid, seed=seed))
            ax.plot(grid, y, color=color, lw=3 if seed == v["improved_seed_used"] else 1.8, ls="-" if seed == v["improved_seed_used"] else "--", label=name)
        ax.scatter([min(v["fertilizer_kg_per_ha"], cap)], [pred], s=110, color="black", zorder=5, label="Your input")
        ax.set_ylim(0, None)
        ax.set_xlabel("Fertilizer (kg per hectare)", fontsize=12)
        ax.set_ylabel("Predicted yield (tons per hectare)", fontsize=12)
        ax.set_title("What if you changed the fertilizer?", fontsize=13, fontweight="bold", loc="left")
        ax.legend(fontsize=10, loc="lower right")
        ax.tick_params(labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        return self._png(fig)

    # ------------------------------------------------------------------------- everything together
    def evaluate_csv(self, csv_data):
        """Run batch evaluation for a CSV containing one plot per row."""
        if isinstance(csv_data, (bytes, bytearray)):
            text = csv_data.decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(text))
        else:
            df = csv_data.copy()

        required = ["region", "crop_type", "survey_year", "planting_month", "altitude_m", "farm_size_ha",
                    "fertilizer_kg_per_ha", "soil_quality_index", "labor_days_per_ha", "distance_to_market_km",
                    "improved_seed_used", "pest_disease_flag"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {"ok": False, "errors": [f"CSV is missing required columns: {', '.join(missing)}."], "warnings": []}

        rows = []
        failures = []
        for idx, row in df.iterrows():
            record = {k: ("" if pd.isna(row[k]) else str(row[k])) for k in required}
            clean, errors, warnings = self.validate(record)
            if errors:
                failures.append({"row": int(idx) + 2, "errors": errors, "warnings": warnings})
                continue
            season_row, info = self.lookup(clean)
            pred = float(self.predict_yield(self.make_rows(clean, season_row))[0])
            revenue = pred * clean["farm_size_ha"] * 10 * info["price"]
            rows.append({
                "row": int(idx) + 2,
                "region": clean["region"],
                "crop_type": clean["crop_type"],
                "survey_year": clean["survey_year"],
                "yield_t_ha": pred,
                "revenue_birr": revenue,
            })

        if not rows:
            return {"ok": False, "errors": ["No valid rows were found in the uploaded CSV."], "warnings": [], "failed_rows": len(failures), "failures": failures}

        avg_yield = float(np.mean([r["yield_t_ha"] for r in rows]))
        avg_revenue = float(np.mean([r["revenue_birr"] for r in rows]))
        summary = {
            "ok": True,
            "errors": [],
            "warnings": [],
            "rows_processed": len(df),
            "valid_rows": len(rows),
            "failed_rows": len(failures),
            "avg_yield_t_ha": avg_yield,
            "avg_revenue_birr": avg_revenue,
            "predictions": rows,
            "failures": failures,
            "summary_line": f"Evaluated {len(rows)} valid plots from {len(df)} rows; average predicted yield {avg_yield:.2f} t/ha and average revenue {avg_revenue:,.0f} birr.",
        }
        return summary

    def run(self, form):
        v, errors, warnings = self.validate(form)
        if errors:
            return {"ok": False, "errors": errors, "warnings": warnings}
        season_row, info = self.lookup(v)
        rows = self.make_rows(v, season_row)
        pred = float(self.predict_yield(rows)[0])
        revenue = pred * v["farm_size_ha"] * 10 * info["price"]
        low = max(pred - self.rmse, 0) * v["farm_size_ha"] * 10 * info["price"]
        high = (pred + self.rmse) * v["farm_size_ha"] * 10 * info["price"]
        estimated = SEASON_LENGTH - info["observed"]
        lookup_line = (f"Growing season {info['window']} {v['survey_year']} in {v['region']}: average temperature {info['temp']:.1f} \u00b0C, "
                       f"station rainfall {info['rain']:.0f} mm, {info['heat']} extreme-heat days"
                       + (f" ({estimated} of {SEASON_LENGTH} months estimated from other years)" if estimated else "")
                       + f". Price for {v['crop_type']} in {v['region']} {v['survey_year']}: {info['price']:,.0f} birr/quintal.")
        return {"ok": True, "errors": [], "warnings": warnings, "inputs": v, "yield_t_ha": pred, "revenue_birr": revenue,
                "revenue_low": low, "revenue_high": high, "rmse": self.rmse, "lookup": info, "lookup_line": lookup_line,
                "chart_compare": self.chart_compare(v, pred), "chart_whatif": self.chart_whatif(v, season_row, pred)}
