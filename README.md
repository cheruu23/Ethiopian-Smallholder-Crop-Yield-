# Ethiopian Smallholder Crop Yield

A research project and Flask web app for predicting smallholder crop yields and
estimating revenue in Ethiopia. The app collects plot details, looks up
seasonal weather and crop prices, then shows a predicted yield, revenue range,
and comparison charts.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app\requirements.txt
python app\train_model.py
python app\app.py
```

Open <http://localhost:7860>. The model builder reads the cleaned training
table in `data/processed/master_train.csv` and the source weather and price
tables in `data/raw/`, then writes the app's model and lookup tables to
`app/assets/`.

## Deploy to Render

The included `render.yaml` is a Render Blueprint. Push this repository to
GitHub, then in Render choose **New > Blueprint** and select the repository.
The service runs from the repository root, installs `app/requirements.txt`,
starts the Flask app with Gunicorn, and checks `/health`. If you already have
a Render service, sync the Blueprint changes and make sure its Root Directory
is blank (repository root), Build Command is `pip install -r
app/requirements.txt`, and Start Command is
`gunicorn --chdir app --workers 1 --threads 4 --timeout 120 app:app`.

The deployment files and app usage are documented in [`app/README.md`](app/README.md).
