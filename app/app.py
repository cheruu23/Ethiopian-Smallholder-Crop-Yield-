"""Crop-yield demo (Flask).   Run:  python app/app.py   then open http://localhost:7860"""
import os

from flask import Flask, jsonify, render_template, request

try:
    from .predictor import CROPS, DEFAULTS, FIELDS, PLANTING_MONTHS, REGIONS, YEARS, Predictor
except ImportError:  # allow running app.py directly or as a module from the project root
    from predictor import CROPS, DEFAULTS, FIELDS, PLANTING_MONTHS, REGIONS, YEARS, Predictor

app = Flask(__name__)
predictor = Predictor()          # loads the bundled tables + model once at start-up


def page(form, result=None):
    return render_template("index.html", form=form, result=result, regions=REGIONS, crops=CROPS, years=YEARS,
                           months=PLANTING_MONTHS, fields=FIELDS, ranges=predictor.ranges, card=predictor.card)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form = request.form.to_dict()
        csv_file = request.files.get("csv_file")
        upload_result = None
        try:
            if csv_file and csv_file.filename:
                if not csv_file.filename.lower().endswith(".csv"):
                    upload_result = {"ok": False, "errors": ["Please upload a CSV file."], "warnings": []}
                else:
                    upload_result = predictor.evaluate_csv(csv_file.read())
            else:
                result = predictor.run(form)
                return page(form, result)
        except Exception:          # last line of defence: never show a stack trace to the user
            app.logger.exception("prediction failed")
            upload_result = {"ok": False, "warnings": [], "errors": ["Sorry, something went wrong with this combination of inputs. Please check the values and try again."]}
        return page(form, upload_result)
    return page(DEFAULTS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Same logic as the form, as JSON (handy for testing)."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "errors": ["Send a JSON object with the plot details."]}), 400
    result = predictor.run({k: ("" if v is None else str(v)) for k, v in payload.items()})
    result.pop("chart_compare", None)
    result.pop("chart_whatif", None)
    return jsonify(result), (200 if result["ok"] else 422)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Open http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
