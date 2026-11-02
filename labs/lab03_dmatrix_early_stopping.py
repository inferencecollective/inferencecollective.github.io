"""Lab 3: Native DMatrix and early stopping validation curves without leakage."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from labs.common import classification_data

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = classification_data(n_samples=640, random_state=42)

    # 1. Outer split to protect test set
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Inner split for early stopping validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool, test_size=0.25, stratify=y_pool, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals_result = {}
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "logloss",
        "eta": 0.06,
        "max_depth": 3,
        "nthread": 1,
        "seed": 42
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=180,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=15,
        evals_result=evals_result,
        verbose_eval=False
    )

    best_iter = booster.best_iteration
    test_preds = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    test_auc = float(roc_auc_score(y_test, test_preds))

    train_loss = evals_result["train"]["logloss"]
    val_loss = evals_result["val"]["logloss"]
    rounds = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(rounds, train_loss, label="Training LogLoss", color="#2563eb", linewidth=1.5)
    ax.plot(rounds, val_loss, label="Validation LogLoss", color="#dc2626", linewidth=2)
    ax.axvline(best_iter + 1, color="#059669", linestyle="--", linewidth=1.5,
               label=f"Best Iteration: {best_iter + 1} (Val Loss: {val_loss[best_iter]:.4f})")

    ax.set_title(f"XGBoost Early Stopping Learning Curves\n(Held-out Test ROC AUC: {test_auc:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Boosting Round (Iteration)")
    ax.set_ylabel("Binary LogLoss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    png_path = output_dir / "03-xgboost-early-stopping.png"
    svg_path = output_dir / "03-xgboost-early-stopping.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)

    metrics = {
        "best_iteration": int(best_iter),
        "val_min_logloss": round(float(val_loss[best_iter]), 4),
        "test_roc_auc": round(test_auc, 3)
    }
    print("Lab 3 Output:", metrics)
    return metrics


if __name__ == "__main__":
    run()
