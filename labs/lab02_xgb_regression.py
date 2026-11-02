"""Lab 2: Nonlinear regression with XGBRegressor, error metrics, and residual diagnostics."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from labs.common import regression_data

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = regression_data(n_samples=400, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    base = DummyRegressor().fit(X_tr, y_tr)
    model = XGBRegressor(
        n_estimators=160,
        max_depth=3,
        learning_rate=0.07,
        tree_method="hist",
        n_jobs=1,
        random_state=42
    ).fit(X_tr, y_tr)

    preds = model.predict(X_te)
    base_preds = base.predict(X_te)

    mae = float(mean_absolute_error(y_te, preds))
    rmse = float(root_mean_squared_error(y_te, preds))
    r2 = float(r2_score(y_te, preds))
    base_rmse = float(root_mean_squared_error(y_te, base_preds))

    residuals = y_te - preds

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    # 1. Actual vs Predicted
    axs[0].scatter(y_te, preds, color="#2563eb", alpha=0.7, edgecolors="none", s=35)
    lims = [min(y_te.min(), preds.min()), max(y_te.max(), preds.max())]
    axs[0].plot(lims, lims, color="#dc2626", linestyle="--", linewidth=1.5, label="Identity line")
    axs[0].set_title(f"Predicted vs Actual (R² = {r2:.3f})", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("Actual Continuous Target")
    axs[0].set_ylabel("Predicted Target")
    axs[0].grid(alpha=0.2)
    axs[0].legend(frameon=False)

    # 2. Residuals vs Predicted
    axs[1].scatter(preds, residuals, color="#059669", alpha=0.7, edgecolors="none", s=35)
    axs[1].axhline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    axs[1].set_title(f"Residual Plot (RMSE = {rmse:.2f} vs Dummy: {base_rmse:.2f})", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Fitted Predictions")
    axs[1].set_ylabel("Residuals (Actual - Predicted)")
    axs[1].grid(alpha=0.2)

    fig.suptitle("XGBRegressor Evaluation and Residual Diagnostics", fontsize=13, fontweight="bold")

    png_path = output_dir / "02-xgboost-regression-residuals.png"
    svg_path = output_dir / "02-xgboost-regression-residuals.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "baseline_rmse": round(base_rmse, 3)
    }
    print("Lab 2 Output:", metrics)
    return metrics


if __name__ == "__main__":
    run()
