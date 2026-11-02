"""Lab 3: ColumnTransformer pipeline with imputation, one-hot encoding, and precision-recall curve."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customers_churn.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run_lab(data_path: Path = DATA_PATH, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    numeric_cols = ["age", "tenure_months", "monthly_charges", "support_tickets"]
    categorical_cols = ["plan", "contract"]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    ap = average_precision_score(y_test, y_proba)
    prevalence = y_test.mean()

    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Average Precision (PR AUC): {ap:.3f} (Prevalence: {prevalence:.2%})")

    # Plot Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    disp = PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name="LogisticRegression Pipeline",
        curve_kwargs=dict(color="#2563eb"),
        ax=ax
    )
    ax.axhline(prevalence, color="#dc2626", linestyle="--", label=f"No-skill Baseline ({prevalence:.1%})")
    ax.set_title(f"Customer Churn Precision-Recall Curve\n(Average Precision: {ap:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Recall (True Positives Captured)")
    ax.set_ylabel("Precision (True Positives / Predicted Positives)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    png_path = output_dir / "03-sklearn-precision-recall.png"
    svg_path = output_dir / "03-sklearn-precision-recall.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    run_lab()
