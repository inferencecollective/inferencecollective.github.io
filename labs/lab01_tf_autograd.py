"""Lab 01: Low-level automatic differentiation with tf.GradientTape.

Fits a linear regression model w*x + b to synthetic data from scratch.
Saves JSON metrics and diagnostic visualizations to artifacts/, outputs/, and previews/.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from labs.common import load_tf_regression, OUTPUTS, PREVIEWS

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    x, y, true_w, true_b = load_tf_regression()

    # Attempt native TensorFlow execution; fall back gracefully to NumPy autograd emulation
    has_tf = False
    try:
        import tensorflow as tf
        has_tf = True
    except ImportError:
        has_tf = False

    epochs = 100
    lr = 0.05
    loss_history = []

    if has_tf:
        w_var = tf.Variable(0.0, dtype=tf.float32)
        b_var = tf.Variable(0.0, dtype=tf.float32)
        x_tf = tf.constant(x, dtype=tf.float32)
        y_tf = tf.constant(y, dtype=tf.float32)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                y_pred = w_var * x_tf + b_var
                loss = tf.reduce_mean(tf.square(y_pred - y_tf))
            loss_history.append(float(loss.numpy()))
            dw, db = tape.gradient(loss, [w_var, b_var])
            w_var.assign_sub(lr * dw)
            b_var.assign_sub(lr * db)

        final_w = float(w_var.numpy())
        final_b = float(b_var.numpy())
        final_loss = float(loss_history[-1])
    else:
        # Exact analytical reverse-mode autodiff matching tf.GradientTape
        w = 0.0
        b = 0.0
        N = len(x)
        for epoch in range(epochs):
            y_pred = w * x + b
            loss = float(np.mean((y_pred - y) ** 2))
            loss_history.append(loss)
            # Gradients: dL/dw = (2/N)*sum((y_pred - y)*x), dL/db = (2/N)*sum(y_pred - y)
            dw = float(2.0 / N * np.sum((y_pred - y) * x))
            db = float(2.0 / N * np.sum(y_pred - y))
            w -= lr * dw
            b -= lr * db

        final_w = float(w)
        final_b = float(b)
        final_loss = float(loss_history[-1])

    # Plot diagnostic figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0b0f19")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values():
            spine.set_color("#374151")

    # Left: Regression fit
    ax1.scatter(x, y, color="#38bdf8", alpha=0.55, s=20, label="Observed data (noise=0.4)")
    x_line = np.linspace(-5.0, 5.0, 100)
    ax1.plot(x_line, true_w * x_line + true_b, color="#34d399", linestyle="--", linewidth=2.0, label=f"Ground truth ({true_w}x + {true_b})")
    ax1.plot(x_line, final_w * x_line + final_b, color="#f43f5e", linewidth=2.2, label=f"Fitted model ({final_w:.2f}x + {final_b:.2f})")
    ax1.set_title("tf.GradientTape Linear Regression Fit", color="#f9fafb", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Input feature x", color="#9ca3af")
    ax1.set_ylabel("Target y", color="#9ca3af")
    ax1.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6")
    ax1.grid(True, color="#1f2937", linestyle=":")

    # Right: Loss trajectory
    ax2.plot(range(1, epochs + 1), loss_history, color="#f59e0b", linewidth=2.2)
    ax2.set_title("MSE Loss Convergence Trajectory", color="#f9fafb", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#9ca3af")
    ax2.set_ylabel("Mean Squared Error", color="#9ca3af")
    ax2.grid(True, color="#1f2937", linestyle=":")

    plt.tight_layout()

    out_png = ARTIFACTS / "01-autograd-regression.png"
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(OUTPUTS / "01-autograd-regression.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(PREVIEWS / "01-autograd-regression.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "lab": "01_tf_autograd",
        "has_tf": has_tf,
        "true_w": true_w,
        "true_b": true_b,
        "final_w": round(final_w, 4),
        "final_b": round(final_b, 4),
        "final_loss": round(final_loss, 4),
        "epochs": epochs,
        "learning_rate": lr,
        "loss_history_first_5": [round(l, 4) for l in loss_history[:5]],
        "loss_history_last_5": [round(l, 4) for l in loss_history[-5:]],
    }

    with open(ARTIFACTS / "01-autograd-regression.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    res = run()
    print(f"Lab 01 completed: w={res['final_w']}, b={res['final_b']}, loss={res['final_loss']}")
