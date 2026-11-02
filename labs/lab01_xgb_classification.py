"""Lab 1: Binary classification with XGBClassifier versus Dummy baseline."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from labs.common import classification_data

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = classification_data(n_samples=500, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    base = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    clf = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=1,
        random_state=42
    ).fit(X_tr, y_tr)

    model_proba = clf.predict_proba(X_te)[:, 1]
    base_proba = base.predict_proba(X_te)[:, 1]

    bal_acc = float(balanced_accuracy_score(y_te, clf.predict(X_te)))
    model_auc = float(roc_auc_score(y_te, model_proba))
    base_auc = float(roc_auc_score(y_te, base_proba))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    # 1. Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        clf, X_te, y_te, cmap="Blues", colorbar=False, ax=axs[0]
    )
    axs[0].set_title(f"Confusion Matrix (Balanced Acc: {bal_acc:.2%})", fontsize=11, fontweight="bold")

    # 2. ROC Curve comparison
    RocCurveDisplay.from_estimator(
        clf, X_te, y_te, name="XGBoost Classifier", curve_kwargs=dict(color="#2563eb", lw=2), ax=axs[1]
    )
    axs[1].plot([0, 1], [0, 1], linestyle=":", color="#9ca3af", label="Chance (AUC = 0.50)")
    axs[1].set_title(f"ROC Curve (Model AUC: {model_auc:.3f} vs Dummy: {base_auc:.3f})", fontsize=11, fontweight="bold")
    axs[1].grid(alpha=0.25)
    axs[1].legend(frameon=False)

    fig.suptitle("XGBoost Binary Classification Evaluation", fontsize=13, fontweight="bold")

    png_path = output_dir / "01-xgboost-classification-roc.png"
    svg_path = output_dir / "01-xgboost-classification-roc.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)

    metrics = {
        "balanced_accuracy": round(bal_acc, 3),
        "model_roc_auc": round(model_auc, 3),
        "baseline_roc_auc": round(base_auc, 3)
    }
    print("Lab 1 Output:", metrics)
    return metrics


if __name__ == "__main__":
    run()
