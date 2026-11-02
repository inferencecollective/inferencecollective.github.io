"""Lab 5: Hyperparameter tuning with manual StratifiedKFold cross-validation and holdout testing."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
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

    # 1. Held-out test vault (20%)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    grid = list(
        ParameterGrid({
            "depth": [3, 5],
            "learning_rate": [0.04, 0.10],
            "l2_leaf_reg": [1, 5],
        })
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_records = []

    for params in grid:
        fold_aucs = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            X_fold_tr, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_fold_tr, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

            train_pool = Pool(X_fold_tr, y_fold_tr, cat_features=CATEGORICAL)
            val_pool = Pool(X_fold_val, y_fold_val, cat_features=CATEGORICAL)

            m = CatBoostClassifier(
                iterations=70,
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                l2_leaf_reg=params["l2_leaf_reg"],
                verbose=False,
                random_seed=42,
                allow_writing_files=False,
            )
            m.fit(train_pool)
            val_probs = m.predict_proba(val_pool)[:, 1]
            fold_aucs.append(roc_auc_score(y_fold_val, val_probs))

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        label = f"depth={params['depth']}, lr={params['learning_rate']}, l2={params['l2_leaf_reg']}"
        cv_records.append({
            "params": params,
            "label": label,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
        })

    cv_df = pd.DataFrame(cv_records).sort_values("mean_auc", ascending=True)
    best_row = cv_df.iloc[-1]
    best_params = best_row["params"]
    best_cv_auc = best_row["mean_auc"]

    # 2. Refit best model on all training data
    full_train_pool = Pool(X_tr, y_tr, cat_features=CATEGORICAL)
    test_pool = Pool(X_te, y_te, cat_features=CATEGORICAL)

    best_model = CatBoostClassifier(
        iterations=90,
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        verbose=False,
        random_seed=42,
        allow_writing_files=False,
    )
    best_model.fit(full_train_pool)

    test_probs = best_model.predict_proba(test_pool)[:, 1]
    test_auc = float(roc_auc_score(y_te, test_probs))

    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    dummy_probs = dummy.predict_proba(X_te)[:, 1]
    dummy_auc = float(roc_auc_score(y_te, dummy_probs))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")

    # Panel 1: Cross-validation comparisons
    axes[0].barh(
        cv_df["label"],
        cv_df["mean_auc"],
        xerr=cv_df["std_auc"],
        color="#fbbf24",
        edgecolor="#d97706",
        alpha=0.85,
    )
    axes[0].set_xlabel("Mean CV ROC AUC (3-Fold)", fontsize=10)
    axes[0].set_xlim(left=max(0.5, cv_df["mean_auc"].min() - 0.08), right=1.0)
    axes[0].set_title(
        f"Manual StratifiedKFold 8-Candidate Grid\n(Best CV AUC: {best_cv_auc:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].grid(True, axis="x", linestyle=":", alpha=0.6)

    # Panel 2: Final Holdout Test ROC
    RocCurveDisplay.from_estimator(
        best_model,
        test_pool,
        y_te,
        ax=axes[1],
        name="Best CatBoost",
        curve_kwargs=dict(color="#f59e0b", lw=2.2),
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
        f"Untouched Holdout Test ROC\n(Test AUC = {test_auc:.3f} vs Dummy {dummy_auc:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].grid(True, linestyle=":", alpha=0.6)

    for target_dir in [output_dir, PREVIEWS_DIR]:
        fig.savefig(target_dir / "05-catboost-cross-validation-roc.png", dpi=160)
        fig.savefig(target_dir / "05-catboost-cross-validation-roc.svg")
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
