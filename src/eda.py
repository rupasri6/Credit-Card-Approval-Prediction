from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import feature_engineering, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


def save_univariate_plots(df):
    columns = ["Time", "Amount", "Amount_Log", "Hour", "Class"]

    for column in columns:
        if column not in df.columns:
            continue

        plt.figure(figsize=(8, 5))
        if column == "Class":
            sns.countplot(data=df, x=column)
            plt.title("Genuine vs Fraud Transactions")
        else:
            sns.histplot(df[column], kde=True)
            plt.title(f"Distribution of {column}")

        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"univariate_{column}.png")
        plt.close()


def save_multivariate_plots(df):
    selected = ["Time", "Amount", "Amount_Log", "Hour", "Class"]
    selected = [column for column in selected if column in df.columns]

    if len(selected) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[selected].corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "multivariate_correlation_heatmap.png")
        plt.close()

    if "Class" in df.columns and "Amount" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Class", y="Amount")
        plt.title("Transaction Amount by Class")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "multivariate_amount_by_class.png")
        plt.close()


def run_eda():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    df = feature_engineering(df)
    save_univariate_plots(df)
    save_multivariate_plots(df)
    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    run_eda()