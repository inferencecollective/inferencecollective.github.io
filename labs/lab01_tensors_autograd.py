"""Lab 01: Dynamic Autograd and Linear Regression in PyTorch."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from labs.common import DATA, OUTPUTS, PREVIEWS


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    data_file = DATA / "pytorch-regression.pt"
    if not data_file.exists():
        from data.generate_samples import generate_all
        generate_all()

    payload = torch.load(str(data_file), weights_only=True)
    x = payload["x"]
    y = payload["y"]
    true_w = payload.get("true_w", 3.5)
    true_b = payload.get("true_b", 1.2)

    # 1. Initialize learnable parameters with requires_grad=True
    torch.manual_seed(42)
    w = torch.nn.Parameter(torch.tensor([[0.0]], dtype=torch.float32))
    b = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    optimizer = torch.optim.SGD([w, b], lr=0.08)

    losses = []
    w_history = []
    b_history = []
    grad_norms = []

    # Initial prediction line
    with torch.no_grad():
        initial_pred = (x * w + b).numpy()

    # 2. Optimization loop
    epochs = 60
    for epoch in range(epochs):
        # Forward pass
        pred = x * w + b
        loss = torch.mean((pred - y) ** 2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Track metrics
        losses.append(loss.item())
        w_history.append(w.item())
        b_history.append(b.item())
        grad_w = w.grad.item() if w.grad is not None else 0.0
        grad_b = b.grad.item() if b.grad is not None else 0.0
        grad_norms.append(np.sqrt(grad_w**2 + grad_b**2))

        # Optimizer step
        optimizer.step()

    final_w = w.item()
    final_b = b.item()
    final_loss = losses[-1]

    with torch.no_grad():
        final_pred = (x * w + b).numpy()

    # 3. Diagnostic 4-panel visual figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="#0f172a")

    # (0, 0) Linear Fit
    ax0 = axes[0, 0]
    ax0.set_facecolor("#1e293b")
    ax0.scatter(x.numpy(), y.numpy(), color="#38bdf8", alpha=0.6, label="Noisy Data (n=100)")
    ax0.plot(x.numpy(), initial_pred, color="#f43f5e", linestyle="--", label=f"Initial (w=0, b=0)")
    ax0.plot(x.numpy(), final_pred, color="#34d399", linewidth=2.5, label=f"Fitted (w={final_w:.2f}, b={final_b:.2f})")
    ax0.set_title("1. Linear Fit & Autograd Regression", color="#f8fafc", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Input x", color="#94a3b8", fontsize=9)
    ax0.set_ylabel("Target y", color="#94a3b8", fontsize=9)
    ax0.tick_params(colors="#94a3b8")
    ax0.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax0.spines.values():
        spine.set_color("#334155")

    # (0, 1) MSE Loss Curve
    ax1 = axes[0, 1]
    ax1.set_facecolor("#1e293b")
    ax1.plot(losses, color="#fde047", linewidth=2, label="Mean Squared Error")
    ax1.set_title(f"2. Loss Convergence (Final: {final_loss:.4f})", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch / Iteration", color="#94a3b8", fontsize=9)
    ax1.set_ylabel("MSE Loss", color="#94a3b8", fontsize=9)
    ax1.set_yscale("log")
    ax1.tick_params(colors="#94a3b8")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # (1, 0) Parameter Trajectories
    ax2 = axes[1, 0]
    ax2.set_facecolor("#1e293b")
    ax2.plot(w_history, color="#38bdf8", linewidth=2, label=f"Learned w (Target: {true_w})")
    ax2.plot(b_history, color="#a78bfa", linewidth=2, label=f"Learned b (Target: {true_b})")
    ax2.axhline(true_w, color="#38bdf8", linestyle=":", alpha=0.6)
    ax2.axhline(true_b, color="#a78bfa", linestyle=":", alpha=0.6)
    ax2.set_title(f"3. Weight Trajectory (w={final_w:.2f}, b={final_b:.2f})", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("Parameter Value", color="#94a3b8", fontsize=9)
    ax2.tick_params(colors="#94a3b8")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # (1, 1) Gradient Magnitude Decay
    ax3 = axes[1, 1]
    ax3.set_facecolor("#1e293b")
    ax3.plot(grad_norms, color="#f43f5e", linewidth=2, label="Gradient Norm ||∇L||")
    ax3.set_title("4. Gradient Norm Decay Across Steps", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax3.set_ylabel("Gradient Norm", color="#94a3b8", fontsize=9)
    ax3.tick_params(colors="#94a3b8")
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax3.spines.values():
        spine.set_color("#334155")

    plt.suptitle("PyTorch Lab 01: Tensors, Autograd & Linear Regression", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "01-tensors-autograd.png"
    preview_file = PREVIEWS / "01-tensors-autograd.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 01 figure to {out_file} and {preview_file}")
    return {
        "final_w": float(final_w),
        "final_b": float(final_b),
        "true_w": float(true_w),
        "true_b": float(true_b),
        "final_loss": float(final_loss),
        "epochs": epochs,
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
