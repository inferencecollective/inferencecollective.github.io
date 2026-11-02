"""Lab 4: Native categorical features, missing value routing, feature gain, and JSON model IO."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from labs.common import churn_data

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = churn_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=19)

    model = XGBClassifier(
        n_estimators=90,
        max_depth=3,
        learning_rate=0.08,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        n_jobs=1,
        random_state=19
    )
    model.fit(X_tr, y_tr)

    # 1. Feature Gain Importance
    booster = model.get_booster()
    score_dict = booster.get_score(importance_type="gain")
    # Map feature names to scores
    feature_names = list(X.columns)
    scores = [score_dict.get(f, 0.0) for f in feature_names]

    # Sort descending
    sorted_idx = np.argsort(scores)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = [scores[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.barh(sorted_features, sorted_scores, color="#3b82f6", edgecolor="#1d4ed8")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_title("XGBoost Native Feature Importance (Average Gain)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Average Gain (Loss Reduction per Split)")
    ax.grid(axis="x", alpha=0.25)

    png_path = output_dir / "04-xgboost-feature-importance.png"
    svg_path = output_dir / "04-xgboost-feature-importance.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)

    # 2. Model JSON Serialization and Reload Verification
    json_path = output_dir / "xgboost_churn_model.json"
    model.save_model(json_path)

    reloaded = XGBClassifier()
    reloaded.load_model(json_path)

    orig_probs = model.predict_proba(X_te)
    reloaded_probs = reloaded.predict_proba(X_te)
    reload_matches = bool(np.allclose(orig_probs, reloaded_probs))
    test_auc = float(roc_auc_score(y_te, orig_probs[:, 1]))

    metrics = {
        "reload_matches": reload_matches,
        "test_roc_auc": round(test_auc, 3),
        "top_feature": sorted_features[-1]
    }
    print("Lab 4 Output:", metrics)
    return metrics


if __name__ == "__main__":
    run()
