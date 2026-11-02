"""Lab 01: ImageFolder Datasets, DataLoaders and Contact Sheet Grids in TorchVision."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from labs.common import DATA, OUTPUTS, PREVIEWS, load_split


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    # 1. Load train split with ImageFolder
    train_dataset = load_split("train")
    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    assert classes == ["diagonal", "horizontal", "vertical"], f"Unexpected classes: {classes}"

    # 2. Mini-batch loader
    batch_size = 12
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))  # images: (12, 3, 64, 64), labels: (12,)

    # 3. Assemble tiled grid with torchvision.utils.make_grid
    grid = make_grid(images, nrow=4, padding=3, pad_value=0.2)  # (3, H_grid, W_grid)

    raw_grid_path = OUTPUTS / "raw_image_grid.png"
    save_image(grid, str(raw_grid_path))

    # 4. Diagnostic Annotated Contact Sheet
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), facecolor="#0f172a")
    for idx, ax in enumerate(axes.flat):
        img_chw = images[idx].numpy()
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        label_idx = labels[idx].item()
        cls_name = classes[label_idx]

        ax.imshow(np.clip(img_hwc, 0.0, 1.0))
        ax.set_title(f"#{idx+1}: {cls_name} (ID: {label_idx})", color="#f8fafc", fontsize=9.5, fontweight="bold")
        ax.set_xlabel(f"Min: {img_hwc.min():.2f} | Max: {img_hwc.max():.2f}", color="#94a3b8", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.5)

    plt.suptitle("TorchVision Lab 01: ImageFolder Ingestion & Annotated Contact Sheet",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "01-imagefolder-grid.png"
    preview_file = PREVIEWS / "01-imagefolder-grid.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 01 figure to {out_file} and {preview_file}")
    return {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "batch_shape": list(images.shape),
        "grid_shape": list(grid.shape),
        "raw_grid_path": str(raw_grid_path),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
