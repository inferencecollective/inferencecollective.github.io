"""Lab 2: Numerical regression with CatBoostRegressor versus Mean Baseline."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = make_friedman1(n_samples=500, noise=1.0, random_state=7)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    reg = CatBoostRegressor(
        iterations=100,
        depth=4,
        learning_rate=0.08,
        verbose=False,
        random_seed=42,
        allow_writing_files=False,
    ).fit(X_tr, y_tr)

    base_preds = base.predict(X_te)
    reg_preds = reg.predict(X_te)
    residuals = y_te - reg_preds

    base_rmse = float(root_mean_squared_error(y_te, base_preds))
    base_mae = float(mean_absolute_error(y_te, base_preds))
    reg_rmse = float(root_mean_squared_error(y_te, reg_preds))
    reg_mae = float(mean_absolute_error(y_te, reg_preds))
    reg_r2 = float(r2_score(y_te, reg_preds))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    # Panel 1: Actual vs Predicted
    axes[0].scatter(y_te, reg_preds, alpha=0.65, color="#f59e0b", edgecolor="white", s=40)
    min_val = min(y_te.min(), reg_preds.min())
    max_val = max(y_te.max(), reg_preds.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", lw=1.8, label="Ideal 45°")
    axes[0].set_xlabel("Actual Values y", fontsize=10)
    axes[0].set_ylabel("Predicted Values ŷ", fontsize=10)
    axes[0].set_title(
        f"Actual vs Predicted (R² = {reg_r2:.3f})\nRMSE: {reg_rmse:.2f} (Baseline: {base_rmse:.2f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper left")

    # Panel 2: Residuals vs Predicted
    axes[1].scatter(reg_preds, residuals, alpha=0.65, color="#06b6d4", edgecolor="white", s=40)
    axes[1].axhline(0, color="#ef4444", linestyle="--", lw=1.8)
    axes[1].set_xlabel("Predicted Values ŷ", fontsize=10)
    axes[1].set_ylabel("Residuals (y - ŷ)", fontsize=10)
    axes[1].set_title(
        f"Residual Diagnostics (MAE = {reg_mae:.2f} vs Base {base_mae:.2f})",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].grid(True, linestyle=":", alpha=0.5)

    for target_dir in [output_dir, PREVIEWS_DIR]:
        fig.savefig(target_dir / "02-catboost-regression-residuals.png", dpi=160)
        fig.savefig(target_dir / "02-catboost-regression-residuals.svg")
    plt.close(fig)

    metrics = {
        "dummy_rmse": round(base_rmse, 2),
        "dummy_mae": round(base_mae, 2),
        "catboost_rmse": round(reg_rmse, 2),
        "catboost_mae": round(reg_mae, 2),
        "catboost_r2": round(reg_r2, 3),
    }
    print(f"Lab 2 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
