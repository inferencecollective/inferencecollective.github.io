"""Lab 3: Native LightGBM Dataset and early stopping without data leakage."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=750,
        n_features=12,
        n_informative=7,
        random_state=9,
    )

    # 1. Held-out test set vault (20%)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Sub-split dev into Train (60% total) and Validation (20% total)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
    )

    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    eval_results = {}
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 12,
        "verbosity": -1,
        "num_threads": 1,
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=180,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=15, verbose=False),
            lgb.record_evaluation(eval_results),
        ],
    )

    best_iter = booster.best_iteration
    test_probs = booster.predict(X_test, num_iteration=best_iter)
    test_auc = float(roc_auc_score(y_test, test_probs))
    test_loss = float(log_loss(y_test, test_probs))

    train_loss = eval_results["train"]["binary_logloss"]
    val_loss = eval_results["valid"]["binary_logloss"]
    rounds = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(rounds, train_loss, label="Training Loss", color="#0284c7", lw=1.8)
    ax.plot(rounds, val_loss, label="Validation Loss", color="#f97316", lw=2.0)
    ax.axvline(best_iter, color="#10b981", linestyle="--", lw=1.8, label=f"Best Round ({best_iter})")
    ax.scatter([best_iter], [val_loss[best_iter - 1]], color="#10b981", s=60, zorder=5)

    ax.set_xlabel("Boosting Rounds", fontsize=10)
    ax.set_ylabel("Binary Logloss", fontsize=10)
    ax.set_title(
        f"LightGBM Early Stopping (Best Round: {best_iter} / 180)\nHeld-Out Test ROC AUC: {test_auc:.3f} | Test Logloss: {test_loss:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    for target_dir in [output_dir, PREVIEWS_DIR]:
        fig.savefig(target_dir / "03-dataset-learning-curves.png", dpi=160)
        fig.savefig(target_dir / "03-dataset-learning-curves.svg")
        fig.savefig(target_dir / "03-lgbm-dataset-early-stopping.png", dpi=160)
        fig.savefig(target_dir / "03-lgbm-dataset-early-stopping.svg")
    plt.close(fig)

    metrics = {
        "best_iteration": int(best_iter),
        "test_auc": round(test_auc, 3),
        "test_loss": round(test_loss, 3),
    }
    print(f"Lab 3 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
