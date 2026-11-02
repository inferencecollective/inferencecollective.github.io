"""Lab 4: Native categorical features, NaN handling, and Booster serialization."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import lightgbm as lgb

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "synthetic_customers.csv"


def load_or_create_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
    else:
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "age": np.random.randint(18, 70, size=n).astype(float),
            "plan": np.random.choice(["basic", "plus", "premium"], size=n),
            "region": np.random.choice(["North", "South", "East", "West"], size=n),
            "monthly_charges": np.random.uniform(20.0, 120.0, size=n),
            "tenure": np.random.randint(1, 60, size=n),
            "churn": np.random.binomial(1, 0.25, size=n),
        })
    df["plan"] = df["plan"].astype("category")
    df["region"] = df["region"].astype("category")
    return df


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_create_data()
    feature_cols = ["age", "plan", "region", "monthly_charges", "tenure"]
    X = df[feature_cols]
    y = df["churn"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    clf = LGBMClassifier(
        n_estimators=60,
        num_leaves=12,
        min_child_samples=15,
        learning_rate=0.06,
        verbosity=-1,
        n_jobs=1,
        random_state=42,
    )
    clf.fit(X_tr, y_tr, categorical_feature=["plan", "region"])

    # 1. Feature Importance by Gain
    booster = clf.booster_
    gains = booster.feature_importance(importance_type="gain")
    gain_series = pd.Series(gains, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = ax.barh(gain_series.index, gain_series.values, color="#0ea5e9", edgecolor="#0284c7")
    ax.set_xlabel("Feature Gain (Objective Improvement)", fontsize=10)
    ax.set_title("LightGBM Feature Importance (Gain)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", fontsize=9)

    for target_dir in [output_dir, PREVIEWS_DIR]:
        fig.savefig(target_dir / "04-categorical-importance.png", dpi=160)
        fig.savefig(target_dir / "04-categorical-importance.svg")
        fig.savefig(target_dir / "04-lgbm-categorical-importance.png", dpi=160)
        fig.savefig(target_dir / "04-lgbm-categorical-importance.svg")
    plt.close(fig)

    # 2. Save and reload model Booster
    model_path = output_dir / "lightgbm_churn_model.txt"
    booster.save_model(str(model_path))

    reloaded_booster = lgb.Booster(model_file=str(model_path))
    probs_orig = clf.predict_proba(X_te)[:, 1]
    probs_reloaded = reloaded_booster.predict(X_te)

    assert np.allclose(probs_orig, probs_reloaded, atol=1e-5), "Reloaded model outputs differ!"
    test_auc = float(roc_auc_score(y_te, probs_orig))

    metrics = {
        "test_auc": round(test_auc, 3),
        "model_file_bytes": model_path.stat().st_size,
        "features_evaluated": len(feature_cols),
    }
    print(f"Lab 4 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
