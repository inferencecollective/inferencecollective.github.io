"""Lab 03: Convolutional Neural Networks (CNN) for Image Classification in PyTorch."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from labs.common import DATA, OUTPUTS, PREVIEWS


class LineCNN(nn.Module):
    """Compact 2D CNN for distinguishing horizontal vs vertical line textures."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        self.fc = nn.Linear(16 * 4 * 4, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.flatten(start_dim=1)
        return self.fc(x)


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    data_file = DATA / "pytorch-lines-16x16.pt"
    if not data_file.exists():
        from data.generate_samples import generate_all
        generate_all()

    payload = torch.load(str(data_file), weights_only=True)
    images = payload["images"]
    labels = payload["labels"]

    # 1. Train / Test split (300 train, 100 test)
    torch.manual_seed(42)
    n_samples = len(images)
    indices = torch.randperm(n_samples)
    n_train = 300
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    img_train, lbl_train = images[train_idx], labels[train_idx]
    img_test, lbl_test = images[test_idx], labels[test_idx]

    train_loader = DataLoader(TensorDataset(img_train, lbl_train), batch_size=32, shuffle=True)

    # 2. Model Initialization & Training
    model = LineCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    epochs = 20
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

        epoch_train_loss = running_loss / len(img_train)
        epoch_train_acc = correct_train / len(img_train)

        model.eval()
        with torch.inference_mode():
            test_logits = model(img_test)
            test_loss = criterion(test_logits, lbl_test).item()
            test_acc = (test_logits.argmax(dim=-1) == lbl_test).float().mean().item()

        train_losses.append(epoch_train_loss)
        test_losses.append(test_loss)
        train_accs.append(epoch_train_acc)
        test_accs.append(test_acc)

    final_test_acc = test_accs[-1]
    num_params = sum(p.numel() for p in model.parameters())

    # 3. Extract Conv1 3x3 filters for visualization
    conv1_weights = model.conv1.weight.detach().cpu().numpy()  # (8, 1, 3, 3)

    # Extract Conv1 feature map for a sample vertical line image
    sample_img = img_test[lbl_test == 1][0:1]
    with torch.inference_mode():
        conv1_feat = model.relu1(model.conv1(sample_img)).squeeze(0).numpy()  # (8, 16, 16)

    # 4. Diagnostic 4-panel visual figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="#0f172a")

    # (0, 0) Sample Synthetic Images: 4 Horizontal (top row) vs 4 Vertical (bottom row)
    ax0 = axes[0, 0]
    ax0.set_facecolor("#1e293b")
    canvas_samples = np.zeros((36, 76), dtype=np.float32)
    h_samples = images[labels == 0][:4].squeeze(1).numpy()
    v_samples = images[labels == 1][:4].squeeze(1).numpy()
    for col_idx in range(4):
        canvas_samples[1:17, 1 + col_idx * 19 : 17 + col_idx * 19] = h_samples[col_idx]
        canvas_samples[19:35, 1 + col_idx * 19 : 17 + col_idx * 19] = v_samples[col_idx]
    ax0.imshow(canvas_samples, cmap="gray")
    ax0.set_title("1. Synthetic 16x16 Patterns (H-Line vs V-Line)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Top: Horiz | Bottom: Vert", color="#94a3b8", fontsize=9)
    ax0.axis("off")

    # (0, 1) Learned Conv1 Filters (8 kernels of 3x3)
    ax1 = axes[0, 1]
    ax1.set_facecolor("#1e293b")
    kernel_canvas = np.zeros((9, 19), dtype=np.float32)
    for k_idx in range(8):
        row_k = (k_idx // 4) * 5 + 1
        col_k = (k_idx % 4) * 5 + 1
        k_val = conv1_weights[k_idx, 0]
        # Normalize filter to [0, 1] for display
        k_norm = (k_val - k_val.min()) / (k_val.max() - k_val.min() + 1e-6)
        kernel_canvas[row_k : row_k + 3, col_k : col_k + 3] = k_norm
    ax1.imshow(kernel_canvas, cmap="viridis")
    ax1.set_title("2. Learned Conv1 Filters (8 x 3x3 Kernels)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # (1, 0) Training & Validation Loss
    ax2 = axes[1, 0]
    ax2.set_facecolor("#1e293b")
    ax2.plot(train_losses, color="#38bdf8", linewidth=2, label="Train Loss")
    ax2.plot(test_losses, color="#fde047", linewidth=2, linestyle="--", label="Test Loss")
    ax2.set_title("3. CNN Cross-Entropy Loss Progression", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("Loss", color="#94a3b8", fontsize=9)
    ax2.tick_params(colors="#94a3b8")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # (1, 1) Conv1 Feature Maps + Metrics
    ax3 = axes[1, 1]
    ax3.set_facecolor("#1e293b")
    feat_canvas = np.zeros((36, 76), dtype=np.float32)
    for f_idx in range(8):
        rf = (f_idx // 4) * 18 + 1
        cf = (f_idx % 4) * 19 + 1
        f_map = conv1_feat[f_idx]
        feat_canvas[rf : rf + 16, cf : cf + 16] = f_map / (f_map.max() + 1e-6)
    ax3.imshow(feat_canvas, cmap="magma")
    ax3.set_title(f"4. Conv1 Feature Maps (Test Acc: {final_test_acc*100:.1f}%)", color="#34d399", fontsize=11, fontweight="bold")
    ax3.axis("off")

    plt.suptitle("PyTorch Lab 03: Convolutional Neural Networks & Feature Maps", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "03-cnn-image-classification.png"
    preview_file = PREVIEWS / "03-cnn-image-classification.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 03 figure to {out_file} and {preview_file}")
    return {
        "final_test_acc": float(final_test_acc),
        "num_parameters": int(num_params),
        "epochs": epochs,
        "sample_shape": list(images.shape),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
