"""Lab 5: Hyperparameter tuning with GridSearchCV and holdout evaluation."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=550,
        n_features=10,
        n_informative=6,
        weights=[0.70, 0.30],
        random_state=5,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_grid = {
        "num_leaves": [7, 15],
        "min_child_samples": [15, 30],
        "learning_rate": [0.03, 0.08],
    }

    base_est = LGBMClassifier(
        n_estimators=60,
        verbosity=-1,
        n_jobs=1,
        random_state=42,
    )

    search = GridSearchCV(
        base_est,
        param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(X_tr, y_tr)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_auc = float(search.best_score_)

    test_probs = best_model.predict_proba(X_te)[:, 1]
    test_auc = float(roc_auc_score(y_te, test_probs))

    # Compile CV results table
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df["label"] = cv_df.apply(
        lambda r: f"leaves={r['param_num_leaves']}, min={r['param_min_child_samples']}, lr={r['param_learning_rate']}",
        axis=1,
    )
    cv_df = cv_df.sort_values("mean_test_score", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")

    # Panel 1: Cross-validation comparisons
    bars = axes[0].barh(
        cv_df["label"],
        cv_df["mean_test_score"],
        xerr=cv_df["std_test_score"],
        color="#38bdf8",
        edgecolor="#0284c7",
        alpha=0.85,
    )
    axes[0].set_xlabel("Mean CV ROC AUC (3-Fold)", fontsize=10)
    axes[0].set_xlim(left=max(0.5, cv_df["mean_test_score"].min() - 0.08), right=1.0)
    axes[0].set_title(
        f"GridSearchCV 8-Configuration Comparison\n(Best CV AUC: {best_cv_auc:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].grid(True, axis="x", linestyle=":", alpha=0.6)

    # Panel 2: Final Holdout Test ROC
    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    RocCurveDisplay.from_estimator(
        best_model,
        X_te,
        y_te,
        ax=axes[1],
        name="Best Tuned LightGBM",
        curve_kwargs=dict(color="#0284c7", lw=2.2),
    )
    RocCurveDisplay.from_estimator(
        dummy,
        X_te,
        y_te,
        ax=axes[1],
        name="Dummy Baseline",
        curve_kwargs=dict(color="#94a3b8", linestyle="--"),
    )
    axes[1].set_title(
        f"Untouched Holdout Test ROC\n(Test AUC = {test_auc:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].grid(True, linestyle=":", alpha=0.6)

    for target_dir in [output_dir, PREVIEWS_DIR]:
        fig.savefig(target_dir / "05-cross-validation-roc.png", dpi=160)
        fig.savefig(target_dir / "05-cross-validation-roc.svg")
        fig.savefig(target_dir / "05-lgbm-tuning-roc.png", dpi=160)
        fig.savefig(target_dir / "05-lgbm-tuning-roc.svg")
    plt.close(fig)

    metrics = {
        "best_params": best_params,
        "best_cv_auc": round(best_cv_auc, 3),
        "test_auc": round(test_auc, 3),
    }
    print(f"Lab 5 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
