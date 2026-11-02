"""Lab 5: Cross-validated hyperparameter tuning on training data only with GridSearchCV."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from labs.common import classification_data

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = classification_data(n_samples=500, random_state=42)

    # Split off test set before any tuning
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    base = XGBClassifier(
        n_estimators=60,
        learning_rate=0.08,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=1,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    param_grid = {
        "max_depth": [2, 4, 6],
        "subsample": [0.7, 0.9, 1.0]
    }

    search = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=1, refit=True)
    search.fit(X_tr, y_tr)

    best_params = search.best_params_
    best_cv_score = float(search.best_score_)
    test_proba = search.predict_proba(X_te)[:, 1]
    test_auc = float(roc_auc_score(y_te, test_proba))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    # 1. Parameter Grid CV Results Heatmap
    depths = param_grid["max_depth"]
    subsamples = param_grid["subsample"]
    scores_matrix = np.zeros((len(depths), len(subsamples)))

    for i, d in enumerate(depths):
        for j, s in enumerate(subsamples):
            mask = [(p["max_depth"] == d and p["subsample"] == s) for p in search.cv_results_["params"]]
            scores_matrix[i, j] = search.cv_results_["mean_test_score"][np.argmax(mask)]

    im = axs[0].imshow(scores_matrix, cmap="YlGnBu", aspect="auto")
    axs[0].set_xticks(range(len(subsamples)), labels=[str(s) for s in subsamples])
    axs[0].set_yticks(range(len(depths)), labels=[str(d) for d in depths])
    axs[0].set_xlabel("subsample")
    axs[0].set_ylabel("max_depth")
    axs[0].set_title(f"3-Fold CV AUC Grid (Best: {best_cv_score:.3f})", fontsize=11, fontweight="bold")

    for i in range(len(depths)):
        for j in range(len(subsamples)):
            axs[0].text(j, i, f"{scores_matrix[i, j]:.3f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=axs[0], label="Mean CV ROC AUC")

    # 2. Final Evaluation on Untouched Test Set
    RocCurveDisplay.from_estimator(
        search.best_estimator_,
        X_te,
        y_te,
        name=f"Best XGBoost (d={best_params['max_depth']}, s={best_params['subsample']})",
        curve_kwargs=dict(color="#059669", lw=2),
        ax=axs[1]
    )
    axs[1].plot([0, 1], [0, 1], linestyle=":", color="#9ca3af", label="Chance (AUC = 0.50)")
    axs[1].set_title(f"Held-Out Test ROC Curve (AUC: {test_auc:.3f})", fontsize=11, fontweight="bold")
    axs[1].grid(alpha=0.25)
    axs[1].legend(frameon=False)

    fig.suptitle("Hyperparameter Grid Search Tuning & Generalization", fontsize=13, fontweight="bold")

    png_path = output_dir / "05-xgboost-grid-search.png"
    svg_path = output_dir / "05-xgboost-grid-search.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)

    metrics = {
        "best_params": best_params,
        "best_cv_auc": round(best_cv_score, 3),
        "test_roc_auc": round(test_auc, 3)
    }
    print("Lab 5 Output:", metrics)
    return metrics


if __name__ == "__main__":
    run()
