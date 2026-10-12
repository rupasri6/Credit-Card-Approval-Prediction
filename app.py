from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "credit_card_model.joblib"

app = Flask(__name__)


def load_bundle():
    return joblib.load(MODEL_PATH)


def to_number(value):
    try:
        return float(value)
    except:
        return 0.0


@app.route("/", methods=["GET", "POST"])
def index():
    bundle = load_bundle()

    if request.method == "POST":
        row = {}

        for feature in bundle["features"]:
            row[feature] = to_number(request.form.get(feature, "0"))

        input_df = pd.DataFrame([row], columns=bundle["features"])
        prediction = int(bundle["model"].predict(input_df)[0])

        probability = None
        if hasattr(bundle["model"], "predict_proba"):
            probability = float(bundle["model"].predict_proba(input_df)[0][prediction])

        result = "Fraud Transaction" if prediction == 1 else "Genuine Transaction"

        return render_template(
            "result.html",
            result=result,
            probability=probability,
            model_name=bundle["model_name"],
        )

    return render_template(
        "index.html",
        features=bundle["features"],
        defaults=bundle["defaults"],
    )


if __name__ == "__main__":
    app.run(debug=True)