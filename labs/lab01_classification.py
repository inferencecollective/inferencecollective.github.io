"""Lab 1: Baseline before complexity - Multi-class Wine classification with confusion matrix."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run_lab(output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_wine(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 1. Trivial Baseline
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))

    # 2. Pipeline with Scaler + Logistic Regression
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    pipe_acc = accuracy_score(y_test, preds)
    pipe_bal_acc = balanced_accuracy_score(y_test, preds)

    print(f"Dummy Accuracy: {dummy_acc:.3f}")
    print(f"Logistic Regression Accuracy: {pipe_acc:.3f} (Balanced: {pipe_bal_acc:.3f})")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    disp = ConfusionMatrixDisplay.from_estimator(
        pipe,
        X_test,
        y_test,
        display_labels=["Class 0", "Class 1", "Class 2"],
        cmap="Blues",
        ax=ax,
        colorbar=False
    )
    ax.set_title(f"Wine 3-Class Confusion Matrix\n(Accuracy: {pipe_acc:.2%}, Baseline: {dummy_acc:.2%})", fontsize=11, fontweight="bold")

    png_path = output_dir / "01-sklearn-confusion-matrix.png"
    svg_path = output_dir / "01-sklearn-confusion-matrix.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


run = run_lab

if __name__ == "__main__":
    run_lab()
