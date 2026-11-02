"""Lab 01: Tensors, Shapes, and Regression with Keras 3 Sequential API.

Trains a Dense regression network on synthetic linear data (y ≈ 2.5x - 0.8 + noise).
Evaluates held-out RMSE and outputs diagnostics to artifacts/, outputs/, and previews/.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from labs.common import load_keras_regression, OUTPUTS, PREVIEWS

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test, true_slope, true_intercept = load_keras_regression()

    has_keras = False
    backend_name = "numpy_fallback"
    try:
        import keras
        from keras import layers
        has_keras = True
        backend_name = keras.backend.backend()
    except Exception:
        has_keras = False

    epochs = 60
    batch_size = 16

    if has_keras:
        model = keras.Sequential([
            keras.Input(shape=(1,)),
            layers.Dense(12, activation="relu"),
            layers.Dense(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")
        hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        train_loss = [float(l) for l in hist.history["loss"]]

        preds_test = model.predict(x_test, verbose=0).ravel()
        test_mse = float(np.mean((preds_test - y_test.ravel()) ** 2))
        test_rmse = float(np.sqrt(test_mse))

        x_line = np.linspace(-3.0, 3.0, 100, dtype=np.float32).reshape(-1, 1)
        preds_line = model.predict(x_line, verbose=0).ravel()
    else:
        # Exact analytical 2-layer MLP in pure NumPy
        np.random.seed(42)
        W1 = np.random.randn(1, 12).astype(np.float32) * np.sqrt(2.0)
        b1 = np.zeros((1, 12), dtype=np.float32)
        W2 = np.random.randn(12, 1).astype(np.float32) * np.sqrt(2.0 / 12)
        b2 = np.zeros((1, 1), dtype=np.float32)

        lr = 0.02
        train_loss = []
        N = len(x_train)

        for ep in range(epochs):
            perm = np.random.permutation(N)
            for s in range(0, N, batch_size):
                b_idx = perm[s:s + batch_size]
                xb, yb = x_train[b_idx], y_train[b_idx]
                z1 = xb @ W1 + b1
                a1 = np.maximum(0, z1)
                pred = a1 @ W2 + b2

                grad_pred = 2.0 * (pred - yb) / len(xb)
                dW2 = a1.T @ grad_pred
                db2 = np.sum(grad_pred, axis=0, keepdims=True)

                da1 = grad_pred @ W2.T
                dz1 = da1 * (z1 > 0)
                dW1 = xb.T @ dz1
                db1 = np.sum(dz1, axis=0, keepdims=True)

                W2 -= lr * dW2
                b2 -= lr * db2
                W1 -= lr * dW1
                b1 -= lr * db1

            cur_pred = np.maximum(0, x_train @ W1 + b1) @ W2 + b2
            train_loss.append(float(np.mean((cur_pred - y_train) ** 2)))

        preds_test = (np.maximum(0, x_test @ W1 + b1) @ W2 + b2).ravel()
        test_mse = float(np.mean((preds_test - y_test.ravel()) ** 2))
        test_rmse = float(np.sqrt(test_mse))

        x_line = np.linspace(-3.0, 3.0, 100, dtype=np.float32).reshape(-1, 1)
        preds_line = (np.maximum(0, x_line @ W1 + b1) @ W2 + b2).ravel()

    # Diagnostic Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0b0f19")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values():
            spine.set_color("#374151")

    # Left: Regression Fit
    ax1.scatter(x_train, y_train, color="#38bdf8", alpha=0.55, s=22, label="Train samples (N=150)")
    ax1.scatter(x_test, y_test, color="#f59e0b", alpha=0.75, s=26, label="Test samples (N=50)")
    ax1.plot(x_line.ravel(), true_slope * x_line.ravel() + true_intercept, color="#34d399",
             linestyle="--", linewidth=2.0, label=f"Ground truth ({true_slope}x + {true_intercept})")
    ax1.plot(x_line.ravel(), preds_line, color="#f43f5e", linewidth=2.2, label="Keras Sequential fit")
    ax1.set_title(f"Keras 3 Regression (Test RMSE: {test_rmse:.3f})", color="#f9fafb", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Input x", color="#9ca3af")
    ax1.set_ylabel("Target y", color="#9ca3af")
    ax1.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6")
    ax1.grid(True, color="#1f2937", linestyle=":")

    # Right: Training Loss Curve
    ax2.plot(range(1, epochs + 1), train_loss, color="#38bdf8", linewidth=2.0)
    ax2.set_title(f"MSE Loss Convergence ({backend_name} backend)", color="#f9fafb", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#9ca3af")
    ax2.set_ylabel("Mean Squared Error", color="#9ca3af")
    ax2.grid(True, color="#1f2937", linestyle=":")

    plt.tight_layout()

    out_png = ARTIFACTS / "01-tensors-regression.png"
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(OUTPUTS / "01-tensors-regression.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(PREVIEWS / "01-tensors-regression.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "lab": "01_tensors_regression",
        "has_keras": has_keras,
        "backend": backend_name,
        "test_rmse": round(test_rmse, 4),
        "test_mse": round(test_mse, 4),
        "true_slope": true_slope,
        "true_intercept": true_intercept,
        "epochs": epochs,
        "final_train_loss": round(train_loss[-1], 4),
    }

    with open(ARTIFACTS / "01-tensors-regression.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    res = run()
    print(f"Lab 01 completed: backend={res['backend']}, test_rmse={res['test_rmse']}")
