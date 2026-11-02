"""Lab 4: Categorical features, missing value imputation, and native CBM model serialization."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from data.generate_dataset import CATEGORICAL, make_customers, prepare_categories

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_categories(make_customers(n=600, random_state=42))
    feature_cols = [c for c in df.columns if c != "churned"]
    X = df[feature_cols]
    y = df["churned"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    train_pool = Pool(X_tr, y_tr, cat_features=CATEGORICAL)
    test_pool = Pool(X_te, y_te, cat_features=CATEGORICAL)

    clf = CatBoostClassifier(
        iterations=90,
        depth=4,
        learning_rate=0.07,
        verbose=False,
        random_seed=42,
        allow_writing_files=False,
    )
    clf.fit(train_pool)

    # 1. Feature Importance
    importances = clf.get_feature_importance()
    feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    bars = ax.barh(feat_series.index, feat_series.values, color="#f59e0b", edgecolor="#d97706")
    ax.set_xlabel("Relative Feature Importance (%)", fontsize=10)
    ax.set_title("CatBoost Feature Importance (PredictionValuesChange)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9)

    for target_dir in [output_dir, PREVIEWS_DIR]:
        fig.savefig(target_dir / "04-catboost-feature-importance.png", dpi=160)
        fig.savefig(target_dir / "04-catboost-feature-importance.svg")
    plt.close(fig)

    # 2. Save and reload native CBM model
    model_path = output_dir / "catboost_churn_model.cbm"
    clf.save_model(str(model_path))

    reloaded = CatBoostClassifier()
    reloaded.load_model(str(model_path))

    # Test on unseen categories and verify predictions match
    unseen_batch = X_te.head(4).copy()
    unseen_batch["region"] = "NewRegion"
    unseen_pool = Pool(unseen_batch, cat_features=CATEGORICAL)

    p_orig = clf.predict_proba(unseen_pool)[:, 1]
    p_reloaded = reloaded.predict_proba(unseen_pool)[:, 1]
    assert np.allclose(p_orig, p_reloaded, atol=1e-5), "Reloaded model outputs differ!"

    test_auc = float(roc_auc_score(y_te, clf.predict_proba(test_pool)[:, 1]))

    metrics = {
        "test_auc": round(test_auc, 3),
        "model_file_bytes": model_path.stat().st_size,
        "features_evaluated": len(feature_cols),
    }
    print(f"Lab 4 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
