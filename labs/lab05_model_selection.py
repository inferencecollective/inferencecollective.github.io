"""Lab 5: Safe hyperparameter tuning - GridSearchCV within Pipeline, validation curves, and test ROC AUC."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run_lab(output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

    search = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", refit=True)
    search.fit(X_train, y_train)

    best_c = search.best_params_["logisticregression__C"]
    best_cv_score = search.best_score_
    test_proba = search.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, test_proba)

    print(f"Best C Parameter: {best_c}")
    print(f"Mean CV ROC AUC: {best_cv_score:.3f}")
    print(f"Held-Out Test ROC AUC: {test_roc_auc:.3f}")

    # Dual plot: CV parameter response and final test ROC curve
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    # 1. Parameter Tuning Curve (Train-only CV)
    means = search.cv_results_["mean_test_score"]
    stds = search.cv_results_["std_test_score"]
    c_vals = [p["logisticregression__C"] for p in search.cv_results_["params"]]

    axs[0].errorbar(c_vals, means, yerr=stds, marker="o", color="#2563eb", capsize=4, linewidth=2)
    axs[0].set_xscale("log")
    axs[0].axvline(best_c, color="#dc2626", linestyle="--", label=f"Best C = {best_c} (CV AUC: {best_cv_score:.3f})")
    axs[0].set_title("5-Fold Cross-Validation Performance (Training Only)", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("Regularization Strength C (log scale)")
    axs[0].set_ylabel("Mean CV ROC AUC")
    axs[0].grid(alpha=0.25)
    axs[0].legend(frameon=False)

    # 2. Final Evaluation on Untouched Test Set
    disp = RocCurveDisplay.from_estimator(
        search.best_estimator_,
        X_test,
        y_test,
        name=f"Tuned Model (C={best_c})",
        curve_kwargs=dict(color="#059669"),
        ax=axs[1]
    )
    axs[1].plot([0, 1], [0, 1], color="#9ca3af", linestyle=":", label="Chance (AUC = 0.50)")
    axs[1].set_title(f"Held-Out Test Set ROC Curve\n(Test ROC AUC: {test_roc_auc:.3f})", fontsize=11, fontweight="bold")
    axs[1].grid(alpha=0.25)
    axs[1].legend(frameon=False)

    fig.suptitle("Model Selection Pipeline: Training CV vs Generalization Test", fontsize=13, fontweight="bold")

    png_path = output_dir / "05-sklearn-model-selection.png"
    svg_path = output_dir / "05-sklearn-model-selection.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    run_lab()
