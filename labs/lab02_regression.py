"""Lab 2: Regression diagnostics - Ridge regression, metrics, and residual distribution."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run_lab(output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = make_regression(n_samples=250, n_features=6, noise=14.0, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Baseline & Ridge Model
    dummy = DummyRegressor().fit(X_train, y_train)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)

    pred = model.predict(X_test)
    dummy_pred = dummy.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    dummy_mae = mean_absolute_error(y_test, dummy_pred)

    print(f"Baseline MAE: {dummy_mae:.2f}")
    print(f"Ridge MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

    residuals = y_test - pred

    # Side-by-side plot: Actual vs Predicted and Residuals vs Predicted
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    # 1. Actual vs Predicted
    axs[0].scatter(y_test, pred, color="#2563eb", alpha=0.7, edgecolors="none", s=35)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    axs[0].plot(lims, lims, color="#dc2626", linestyle="--", linewidth=1.5, label="Identity line")
    axs[0].set_title(f"Predicted vs Actual (R² = {r2:.3f})", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("Actual Values")
    axs[0].set_ylabel("Predicted Values")
    axs[0].grid(alpha=0.2)
    axs[0].legend(frameon=False)

    # 2. Residuals vs Predicted
    axs[1].scatter(pred, residuals, color="#059669", alpha=0.7, edgecolors="none", s=35)
    axs[1].axhline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    axs[1].set_title(f"Residual Plot (MAE = {mae:.2f}, RMSE = {rmse:.2f})", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Fitted Predictions")
    axs[1].set_ylabel("Residuals (Actual - Predicted)")
    axs[1].grid(alpha=0.2)

    fig.suptitle("Ridge Regression Evaluation and Diagnostic Checks", fontsize=13, fontweight="bold")

    png_path = output_dir / "02-sklearn-regression-residuals.png"
    svg_path = output_dir / "02-sklearn-regression-residuals.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


run = run_lab

if __name__ == "__main__":
    run_lab()
