---
title: Crop Yield and Revenue Estimator
emoji: 🌾
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Crop Yield & Revenue Estimator (Deliverable E)

A small Flask web app. Enter a plot's details, get a **predicted yield (tons/ha)** and an **estimated revenue (birr)**.
You never type weather or prices: the app looks them up from the cleaned tables in `assets/`.

## Run it locally

```bash
pip install -r app/requirements.txt
python app/train_model.py
python app/app.py
```

Open http://localhost:7860. The training script builds the app's lookup tables
and model from `data/processed/master_train.csv` and the source tables under
`data/raw/`.

## What the user does

| Input | How |
|---|---|
| Region, crop type, survey year (2021-2024), planting month (Feb, Mar, Jun, Jul, Aug) | drop-down lists |
| Altitude (m), farm size (ha), fertilizer (kg/ha), soil quality (0-1), labour (days/ha), distance to market (km) | numbers |
| Improved seed, pest/disease problem | Yes / No |

## What the app looks up (automatically)

* **Weather:** `assets/weather_clean.csv` (monthly, cleaned in notebook 01). Growing season = planting month + next 3 months. The app computes the season mean temperature, rainfall total, extreme-heat days and the temperature deviation from the region's typical season for that planting month: the same function and the same numbers as the master table (checked in `test_app.py`).
* **Price:** `assets/prices_clean.csv` (unit-corrected birr per quintal) for the chosen crop + region + year.

## Outputs

* Predicted yield (t/ha) with the typical error (cross-validated RMSE).
* Estimated revenue = predicted yield x farm size x 10 x price, with a likely range (yield +/- one RMSE).
* A line showing the looked-up weather and price, e.g. *"Growing season Jun-Sep 2023 in Amhara: average temperature 15.2 °C, station rainfall 709 mm, 0 extreme-heat days. Price for maize in Amhara 2023: 2,978 birr/quintal."*

* Two charts: your prediction vs the average yield for that region and crop, and a what-if chart of yield across fertilizer amounts (with and without improved seed).

## Bad input

Empty, non-numeric, negative, absurd or tampered values give a red list of plain-language messages (nothing crashes, no stack traces). Valid values outside the range the model learned from still get an answer, with a yellow warning.

## The model

HistGradientBoosting (scikit-learn) trained on the cleaned plots in notebook 01 (15,090 rows in the current training table). 19 inputs: region, crop, the nine plot details, season weather (4 features), planting month (2) and two engineered features (fertilizer x improved seed, aridity ratio). Plot-reported rainfall is **not** used, because the form does not ask for it and it carries almost no signal (notebook 02, B4.3). Numbers are in `assets/model_card.json`; retrain with `python app/train_model.py` (needs `data/processed/` from notebook 01). If `model.joblib` does not load (different scikit-learn version) the app retrains itself from `assets/training_data.csv` in a few seconds.

## Deploy to Render

The project root contains `render.yaml` for a Render Blueprint. Push this
repository to GitHub, create a Blueprint in Render, and select the repository.
Render installs this folder's dependencies, starts the Flask app with Gunicorn,
and checks the `/health` endpoint. App assets are stored in `assets/`; run
`python app/train_model.py` from the project root to rebuild them.

## Files

```
app/
  app.py            web routes (form page, /api/predict, /health)
  predictor.py      validation, weather/price lookup, prediction, charts
  train_model.py    builds app assets from project data
  test_app.py       Flask route smoke tests
  templates/index.html
  assets/           weather_clean.csv, prices_clean.csv, training_data.csv, model.joblib, model_card.json
  requirements.txt  pinned runtime dependencies
```
