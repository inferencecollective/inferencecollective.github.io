"""Lab 3: Native CatBoost Pool and early stopping without data leakage."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from data.generate_dataset import CATEGORICAL, make_customers, prepare_categories

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_categories(make_customers(n=600, random_state=42))
    X = df.drop(columns="churned")
    y = df["churned"]

    # 1. 20% Held-out test vault
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. 80% Dev sub-split into Train (60% total) and Val (20% total)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
    )

    train_pool = Pool(X_tr, y_tr, cat_features=CATEGORICAL)
    val_pool = Pool(X_val, y_val, cat_features=CATEGORICAL)
    test_pool = Pool(X_test, y_test, cat_features=CATEGORICAL)

    model = CatBoostClassifier(
        iterations=160,
        depth=4,
        learning_rate=0.06,
        eval_metric="Logloss",
        verbose=False,
        random_seed=42,
        allow_writing_files=False,
    )
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=18,
        use_best_model=True,
    )

    best_iter = model.get_best_iteration()
    evals = model.get_evals_result()
    train_loss = evals["learn"]["Logloss"]
    val_loss = evals["validation"]["Logloss"]
    rounds = range(1, len(train_loss) + 1)

    test_probs = model.predict_proba(test_pool)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_probs))
    test_loss_val = float(log_loss(y_test, test_probs))

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(rounds, train_loss, label="Training Loss", color="#0284c7", lw=1.8)
    ax.plot(rounds, val_loss, label="Validation Loss", color="#f59e0b", lw=2.0)
    ax.axvline(best_iter + 1, color="#10b981", linestyle="--", lw=1.8, label=f"Best Iteration ({best_iter + 1})")
    ax.scatter([best_iter + 1], [val_loss[best_iter]], color="#10b981", s=60, zorder=5)

    ax.set_xlabel("Boosting Rounds", fontsize=10)
    ax.set_ylabel("Logloss", fontsize=10)
    ax.set_title(
        f"CatBoost Early Stopping Learning Curves (Best Iteration: {best_iter + 1})\nHeld-Out Test ROC AUC: {test_auc:.3f} | Test Logloss: {test_loss_val:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    for target_dir in [output_dir, PREVIEWS_DIR]:
        fig.savefig(target_dir / "03-catboost-learning-curves.png", dpi=160)
        fig.savefig(target_dir / "03-catboost-learning-curves.svg")
    plt.close(fig)

    metrics = {
        "best_iteration": int(best_iter),
        "test_auc": round(test_auc, 3),
        "test_loss": round(test_loss_val, 3),
    }
    print(f"Lab 3 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
