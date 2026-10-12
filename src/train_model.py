from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing import load_dataset, split_features_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def train_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
    }

    results = []
    best_score = -1
    best_name = ""
    best_pipeline = None

    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        results.append(
            {
                "model": name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        report_path = REPORT_DIR / f"{name.lower().replace(' ', '_')}_report.txt"
        report_path.write_text(
            f"{classification_report(y_test, predictions, zero_division=0)}\n\n"
            f"Confusion Matrix\n{confusion_matrix(y_test, predictions)}\n",
            encoding="utf-8",
        )

        if f1 > best_score:
            best_score = f1
            best_name = name
            best_pipeline = pipeline

    pd.DataFrame(results).sort_values("f1_score", ascending=False).to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    bundle = {
        "model": best_pipeline,
        "model_name": best_name,
        "features": x.columns.tolist(),
        "defaults": x.median(numeric_only=True).to_dict(),
    }

    joblib.dump(bundle, MODEL_DIR / "credit_card_model.joblib")

    print(f"Best model: {best_name}")
    print(f"Saved model to: {MODEL_DIR / 'credit_card_model.joblib'}")


if __name__ == "__main__":
    train_models()