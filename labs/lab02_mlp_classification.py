"""Lab 02: Multi-Layer Perceptron (MLP) Non-Linear Decision Boundaries in PyTorch."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from labs.common import DATA, OUTPUTS, PREVIEWS


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    data_file = DATA / "pytorch-moons.pt"
    if not data_file.exists():
        from data.generate_samples import generate_all
        generate_all()

    payload = torch.load(str(data_file), weights_only=True)
    X = payload["X"]
    y = payload["y"]

    # 1. Train / Test split (70% train, 30% test) without leakage
    torch.manual_seed(42)
    n_samples = len(X)
    indices = torch.randperm(n_samples)
    n_train = int(n_samples * 0.7)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    # 2. MLP Architecture
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    # 3. Training & Evaluation loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            correct_train += (logits.argmax(dim=-1) == batch_y).sum().item()

        epoch_train_loss = running_loss / len(X_train)
        epoch_train_acc = correct_train / len(X_train)

        # Evaluation mode
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test)
            test_loss = criterion(test_logits, y_test).item()
            test_acc = (test_logits.argmax(dim=-1) == y_test).float().mean().item()

        train_losses.append(epoch_train_loss)
        test_losses.append(test_loss)
        train_accs.append(epoch_train_acc)
        test_accs.append(test_acc)

    # 4. Meshgrid for Decision Boundary visualization
    x_min, x_max = X[:, 0].min().item() - 0.5, X[:, 0].max().item() + 0.5
    y_min, y_max = X[:, 1].min().item() - 0.5, X[:, 1].max().item() + 0.5
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid_tensor = torch.tensor(np.c_[grid_x.ravel(), grid_y.ravel()], dtype=torch.float32)

    model.eval()
    with torch.inference_mode():
        grid_probs = torch.softmax(model(grid_tensor), dim=-1)[:, 1].numpy().reshape(grid_x.shape)
        final_test_preds = model(X_test).argmax(dim=-1).numpy()

    # Confusion matrix on test split
    y_test_np = y_test.numpy()
    tp = int(((final_test_preds == 1) & (y_test_np == 1)).sum())
    tn = int(((final_test_preds == 0) & (y_test_np == 0)).sum())
    fp = int(((final_test_preds == 1) & (y_test_np == 0)).sum())
    fn = int(((final_test_preds == 0) & (y_test_np == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])

    # 5. Diagnostic 4-panel visual figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="#0f172a")

    # (0, 0) Decision Boundary
    ax0 = axes[0, 0]
    ax0.set_facecolor("#1e293b")
    contour = ax0.contourf(grid_x, grid_y, grid_probs, levels=25, cmap="coolwarm", alpha=0.75)
    ax0.contour(grid_x, grid_y, grid_probs, levels=[0.5], colors="#f8fafc", linewidths=2)
    # Scatter test points
    c0 = y_test_np == 0
    c1 = y_test_np == 1
    ax0.scatter(X_test[c0, 0], X_test[c0, 1], color="#38bdf8", edgecolor="#0f172a", label="Class 0 (Test)", s=50)
    ax0.scatter(X_test[c1, 0], X_test[c1, 1], color="#f43f5e", edgecolor="#0f172a", label="Class 1 (Test)", s=50)
    ax0.set_title(f"1. Non-Linear Decision Boundary (Test Acc: {test_accs[-1]*100:.1f}%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax0.tick_params(colors="#94a3b8")
    ax0.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax0.spines.values():
        spine.set_color("#334155")

    # (0, 1) Train vs Test Loss
    ax1 = axes[0, 1]
    ax1.set_facecolor("#1e293b")
    ax1.plot(train_losses, color="#38bdf8", linewidth=2, label="Train Loss")
    ax1.plot(test_losses, color="#fde047", linewidth=2, linestyle="--", label="Test Loss")
    ax1.set_title("2. Cross-Entropy Loss Curves", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax1.set_ylabel("Loss", color="#94a3b8", fontsize=9)
    ax1.tick_params(colors="#94a3b8")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # (1, 0) Train vs Test Accuracy
    ax2 = axes[1, 0]
    ax2.set_facecolor("#1e293b")
    ax2.plot([a * 100 for a in train_accs], color="#34d399", linewidth=2, label="Train Accuracy")
    ax2.plot([a * 100 for a in test_accs], color="#a78bfa", linewidth=2, linestyle="--", label="Test Accuracy")
    ax2.set_title(f"3. Generalization Accuracy (Final Test: {test_accs[-1]*100:.1f}%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("Accuracy (%)", color="#94a3b8", fontsize=9)
    ax2.tick_params(colors="#94a3b8")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # (1, 1) Confusion Matrix
    ax3 = axes[1, 1]
    ax3.set_facecolor("#1e293b")
    im = ax3.imshow(cm, cmap="Blues")
    ax3.set_title("4. Test Split Confusion Matrix", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Pred Class 0", "Pred Class 1"], color="#94a3b8")
    ax3.set_yticklabels(["True Class 0", "True Class 1"], color="#94a3b8")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center", color="#0f172a" if cm[i, j] > len(y_test)//4 else "#f8fafc", fontsize=12, fontweight="bold")
    for spine in ax3.spines.values():
        spine.set_color("#334155")

    plt.suptitle("PyTorch Lab 02: Multi-Layer Perceptron & Decision Boundaries", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "02-mlp-classification.png"
    preview_file = PREVIEWS / "02-mlp-classification.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 02 figure to {out_file} and {preview_file}")
    return {
        "final_train_acc": float(train_accs[-1]),
        "final_test_acc": float(test_accs[-1]),
        "final_test_loss": float(test_losses[-1]),
        "epochs": epochs,
        "confusion_matrix": cm.tolist(),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
