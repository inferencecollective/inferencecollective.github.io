"""Lab 03: Train and Evaluate TinyVisionCNN with ImageFolder in TorchVision."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from labs.common import DATA, OUTPUTS, PREVIEWS, load_split


class TinyVisionCNN(nn.Module):
    """Compact 2D CNN architecture for classifying 64x64 synthetic shape patterns."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    train_ds = load_split("train", augment=True)
    val_ds = load_split("val", augment=False)
    test_ds = load_split("test", augment=False)
    classes = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=12, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=12, shuffle=False, num_workers=0)

    # 2. Model & Optimizer
    torch.manual_seed(42)
    model = TinyVisionCNN(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 18
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_path = OUTPUTS / "tinyvision-cnn.pt"

    # 3. Training Loop with Validation Checkpointing
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            correct_train += (logits.argmax(dim=-1) == batch_y).sum().item()
            total_train += len(batch_y)

        tr_loss = running_loss / total_train
        tr_acc = correct_train / total_train

        # Validation evaluation
        model.eval()
        val_loss_sum = 0.0
        correct_val = 0
        total_val = 0
        with torch.inference_mode():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                val_loss_sum += criterion(logits, batch_y).item() * len(batch_y)
                correct_val += (logits.argmax(dim=-1) == batch_y).sum().item()
                total_val += len(batch_y)

        val_loss = val_loss_sum / total_val
        val_acc = correct_val / total_val

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), str(checkpoint_path))

    # 4. Final Held-Out Test Evaluation
    # Load best checkpoint with weights_only=True
    model.load_state_dict(torch.load(str(checkpoint_path), weights_only=True, map_location="cpu"))
    model.eval()

    all_preds, all_targets = [], []
    with torch.inference_mode():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.numpy().tolist())
            all_targets.extend(batch_y.numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = float((all_preds == all_targets).mean())

    # Confusion matrix
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    # Per-class metrics
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_precision.append(round(prec, 3))
        per_class_recall.append(round(rec, 3))
        per_class_f1.append(round(f1, 3))

    metadata = {
        "model": "TinyVisionCNN",
        "num_classes": num_classes,
        "classes": classes,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "final_test_accuracy": round(test_acc, 4),
        "per_class_precision": dict(zip(classes, per_class_precision)),
        "per_class_recall": dict(zip(classes, per_class_recall)),
        "per_class_f1": dict(zip(classes, per_class_f1)),
    }
    meta_path = artifacts_dir / "tinyvision-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # 5. Diagnostic 4-panel visual figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="#0f172a")

    # (0, 0) Loss Curves
    ax0 = axes[0, 0]
    ax0.set_facecolor("#1e293b")
    ax0.plot(train_losses, color="#38bdf8", linewidth=2, label="Train Loss")
    ax0.plot(val_losses, color="#fde047", linewidth=2, linestyle="--", label="Val Loss")
    ax0.axvline(best_epoch - 1, color="#34d399", linestyle=":", label=f"Best Ep #{best_epoch}")
    ax0.set_title("1. Cross-Entropy Loss Progression", color="#f8fafc", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax0.set_ylabel("Loss", color="#94a3b8", fontsize=9)
    ax0.tick_params(colors="#94a3b8")
    ax0.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax0.spines.values():
        spine.set_color("#334155")

    # (0, 1) Accuracy Curves
    ax1 = axes[0, 1]
    ax1.set_facecolor("#1e293b")
    ax1.plot([a * 100 for a in train_accs], color="#34d399", linewidth=2, label="Train Acc")
    ax1.plot([a * 100 for a in val_accs], color="#a78bfa", linewidth=2, linestyle="--", label="Val Acc")
    ax1.set_title(f"2. Accuracy Curves (Final Test: {test_acc*100:.1f}%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax1.set_ylabel("Accuracy (%)", color="#94a3b8", fontsize=9)
    ax1.tick_params(colors="#94a3b8")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # (1, 0) Confusion Matrix
    ax2 = axes[1, 0]
    ax2.set_facecolor("#1e293b")
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_title("3. Held-Out Test Set Confusion Matrix", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(num_classes))
    ax2.set_yticks(range(num_classes))
    ax2.set_xticklabels(classes, color="#94a3b8", fontsize=9)
    ax2.set_yticklabels(classes, color="#94a3b8", fontsize=9)
    for i in range(num_classes):
        for j in range(num_classes):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="#0f172a" if cm[i, j] > len(all_targets)//num_classes//2 else "#f8fafc",
                     fontweight="bold")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # (1, 1) Per-Class Metrics
    ax3 = axes[1, 1]
    ax3.set_facecolor("#1e293b")
    x_pos = np.arange(num_classes)
    width = 0.25
    ax3.bar(x_pos - width, per_class_precision, width, label="Precision", color="#38bdf8")
    ax3.bar(x_pos, per_class_recall, width, label="Recall", color="#34d399")
    ax3.bar(x_pos + width, per_class_f1, width, label="F1-Score", color="#fde047")
    ax3.set_title("4. Test Split Per-Class Metrics", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(classes, color="#94a3b8", fontsize=9)
    ax3.tick_params(colors="#94a3b8")
    ax3.set_ylim([0, 1.15])
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax3.spines.values():
        spine.set_color("#334155")

    plt.suptitle("TorchVision Lab 03: TinyVisionCNN Training & Validation Lifecycle",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "03-classifier-training.png"
    preview_file = PREVIEWS / "03-classifier-training.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 03 figure to {out_file} and metadata to {meta_path}")
    return {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "test_acc": float(test_acc),
        "metadata_path": str(meta_path),
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
