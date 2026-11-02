"""Lab 1: Binary classification with CatBoostClassifier versus Dummy baseline."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        weights=[0.70, 0.30],
        random_state=42,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    base = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    clf = CatBoostClassifier(
        iterations=80,
        depth=4,
        learning_rate=0.08,
        verbose=False,
        random_seed=42,
        allow_writing_files=False,
    ).fit(X_tr, y_tr)

    base_probs = base.predict_proba(X_te)[:, 1]
    clf_probs = clf.predict_proba(X_te)[:, 1]
    clf_preds = clf.predict(X_te)

    base_auc = float(roc_auc_score(y_te, base_probs))
    clf_auc = float(roc_auc_score(y_te, clf_probs))
    clf_acc = float(accuracy_score(y_te, clf_preds))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")

    # Panel 1: ROC Curves
    RocCurveDisplay.from_estimator(
        clf,
        X_te,
        y_te,
        ax=axes[0],
        name="CatBoost",
        curve_kwargs=dict(color="#f59e0b", lw=2.2),
    )
    RocCurveDisplay.from_estimator(
        base,
        X_te,
        y_te,
        ax=axes[0],
        name="Dummy Baseline",
        curve_kwargs=dict(color="#94a3b8", linestyle="--"),
    )
    axes[0].set_title(
        f"CatBoost ROC Curve (AUC = {clf_auc:.3f} vs Dummy {base_auc:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_te,
        clf_preds,
        display_labels=["Class 0", "Class 1"],
        cmap="YlOrBr",
        ax=axes[1],
        colorbar=False,
    )
    axes[1].set_title(
        f"Confusion Matrix (Acc = {clf_acc:.1%})",
        fontsize=11,
        fontweight="bold",
    )

    for target_dir in [output_dir, PREVIEWS_DIR]:
        fig.savefig(target_dir / "01-catboost-classification-roc.png", dpi=160)
        fig.savefig(target_dir / "01-catboost-classification-roc.svg")
    plt.close(fig)

    metrics = {
        "dummy_auc": round(base_auc, 3),
        "catboost_auc": round(clf_auc, 3),
        "catboost_accuracy": round(clf_acc, 3),
    }
    print(f"Lab 1 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
