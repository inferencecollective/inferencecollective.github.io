"""Lab 02: Binary classification with Keras Sequential API on non-linear data.

Trains a multi-layer perceptron on synthetic 2D moons and visualizes decision boundaries.
Saves JSON metrics and figures to artifacts/, outputs/, and previews/.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from labs.common import load_tf_moons, OUTPUTS, PREVIEWS

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_tf_moons()

    has_tf = False
    try:
        import tensorflow as tf
        has_tf = True
    except ImportError:
        has_tf = False

    epochs = 60
    train_loss_hist = []
    val_loss_hist = []

    if has_tf:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        train_loss_hist = [float(x) for x in hist.history["loss"]]
        val_loss_hist = [float(x) for x in hist.history["val_loss"]]
        eval_res = model.evaluate(X_test, y_test, verbose=0)
        test_loss = float(eval_res[0])
        test_acc = float(eval_res[1])
        predict_fn = lambda x: model.predict(x, verbose=0)
    else:
        # High-performance 2-layer MLP with Adam optimizer in NumPy
        np.random.seed(42)
        W1 = np.random.randn(2, 16).astype(np.float32) * np.sqrt(2.0 / 2)
        b1 = np.zeros((1, 16), dtype=np.float32)
        W2 = np.random.randn(16, 8).astype(np.float32) * np.sqrt(2.0 / 16)
        b2 = np.zeros((1, 8), dtype=np.float32)
        W3 = np.random.randn(8, 1).astype(np.float32) * np.sqrt(2.0 / 8)
        b3 = np.zeros((1, 1), dtype=np.float32)

        # Adam moment buffers
        mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
        mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
        mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
        mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
        mW3, vW3 = np.zeros_like(W3), np.zeros_like(W3)
        mb3, vb3 = np.zeros_like(b3), np.zeros_like(b3)

        lr = 0.02
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0
        batch_size = 32
        N = len(X_train)

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))

        def forward(X):
            z1 = X @ W1 + b1
            a1 = np.maximum(0, z1)
            z2 = a1 @ W2 + b2
            a2 = np.maximum(0, z2)
            z3 = a2 @ W3 + b3
            a3 = sigmoid(z3)
            return z1, a1, z2, a2, z3, a3

        def compute_bce(preds, targets):
            p = np.clip(preds, 1e-7, 1.0 - 1e-7)
            t = targets.reshape(-1, 1)
            return float(-np.mean(t * np.log(p) + (1.0 - t) * np.log(1.0 - p)))

        for epoch in range(epochs):
            indices = np.random.permutation(N)
            for start_idx in range(0, N, batch_size):
                t_step += 1
                b_idx = indices[start_idx:start_idx + batch_size]
                xb, yb = X_train[b_idx], y_train[b_idx].reshape(-1, 1)

                z1, a1, z2, a2, z3, a3 = forward(xb)
                # Backward pass
                dz3 = (a3 - yb) / len(xb)
                dW3 = a2.T @ dz3
                db3 = np.sum(dz3, axis=0, keepdims=True)

                da2 = dz3 @ W3.T
                dz2 = da2 * (z2 > 0)
                dW2 = a1.T @ dz2
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = dz2 @ W2.T
                dz1 = da1 * (z1 > 0)
                dW1 = xb.T @ dz1
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # Adam updates
                for param, grad, m, v in [
                    (W1, dW1, mW1, vW1), (b1, db1, mb1, vb1),
                    (W2, dW2, mW2, vW2), (b2, db2, mb2, vb2),
                    (W3, dW3, mW3, vW3), (b3, db3, mb3, vb3)
                ]:
                    m[:] = beta1 * m + (1 - beta1) * grad
                    v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
                    m_hat = m / (1 - beta1 ** t_step)
                    v_hat = v / (1 - beta2 ** t_step)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # Record epoch losses
            _, _, _, _, _, tr_preds = forward(X_train)
            _, _, _, _, _, te_preds = forward(X_test)
            train_loss_hist.append(compute_bce(tr_preds, y_train))
            val_loss_hist.append(compute_bce(te_preds, y_test))

        _, _, _, _, _, final_test_preds = forward(X_test)
        test_loss = compute_bce(final_test_preds, y_test)
        test_acc = float(np.mean((final_test_preds.ravel() >= 0.5) == y_test))

        def predict_fn(X):
            _, _, _, _, _, p = forward(X)
            return p

    # Plot diagnostic figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0b0f19")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values():
            spine.set_color("#374151")

    # Left: Decision boundary
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs = predict_fn(grid_points).reshape(xx.shape)

    contour = ax1.contourf(xx, yy, probs, levels=20, cmap="viridis", alpha=0.85)
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.ax.tick_params(colors="#9ca3af")
    cbar.set_label("Predicted P(class=1)", color="#9ca3af")

    # Plot test points
    mask0 = (y_test == 0)
    mask1 = (y_test == 1)
    ax1.scatter(X_test[mask0, 0], X_test[mask0, 1], c="#38bdf8", edgecolors="#ffffff", s=40, label="Class 0 (Moon A)")
    ax1.scatter(X_test[mask1, 0], X_test[mask1, 1], c="#f43f5e", edgecolors="#ffffff", s=40, label="Class 1 (Moon B)")
    ax1.set_title(f"Keras Decision Boundary (Acc: {test_acc:.1%})", color="#f9fafb", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Feature X1", color="#9ca3af")
    ax1.set_ylabel("Feature X2", color="#9ca3af")
    ax1.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6", loc="upper right")

    # Right: Loss curves
    ax2.plot(range(1, epochs + 1), train_loss_hist, color="#38bdf8", label="Train Loss", linewidth=2.0)
    ax2.plot(range(1, epochs + 1), val_loss_hist, color="#34d399", label="Val Loss", linewidth=2.0)
    ax2.set_title("Binary Crossentropy Loss Curves", color="#f9fafb", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#9ca3af")
    ax2.set_ylabel("Loss", color="#9ca3af")
    ax2.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6")
    ax2.grid(True, color="#1f2937", linestyle=":")

    plt.tight_layout()

    out_png = ARTIFACTS / "02-keras-classification.png"
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(OUTPUTS / "02-keras-classification.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(PREVIEWS / "02-keras-classification.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "lab": "02_keras_classification",
        "has_tf": has_tf,
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "epochs": epochs,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "final_train_loss": round(train_loss_hist[-1], 4),
        "final_val_loss": round(val_loss_hist[-1], 4),
    }

    with open(ARTIFACTS / "02-keras-classification.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    res = run()
    print(f"Lab 02 completed: test_acc={res['test_accuracy']}, test_loss={res['test_loss']}")
