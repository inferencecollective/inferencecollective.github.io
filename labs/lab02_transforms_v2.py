"""Lab 02: Visualizing torchvision.transforms.v2 Preprocessing & Augmentation Pipelines."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from labs.common import DATA, OUTPUTS, PREVIEWS


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    sample_path = DATA / "synthetic_shapes" / "train" / "diagonal" / "shape_000.png"
    if not sample_path.exists():
        from data.generate_samples import generate_all
        generate_all()

    pil_img = Image.open(sample_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # 1. Base conversion (PIL -> Image Tensor -> Float32 [0.0, 1.0])
    base_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    tensor_base = base_tf(pil_img)  # (3, 64, 64)

    # 2. Resizing
    resize_tf = v2.Resize((32, 32))
    tensor_resized = resize_tf(tensor_base)

    # 3. Deterministic Rotation
    rotate_tf = v2.RandomRotation(degrees=(30, 30))
    tensor_rotated = rotate_tf(tensor_base)

    # 4. Forced Horizontal Flip (demonstrating why orientation matters!)
    flip_tf = v2.RandomHorizontalFlip(p=1.0)
    tensor_flipped = flip_tf(tensor_base)

    # 5. Color Jitter
    jitter_tf = v2.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.5)
    tensor_jittered = jitter_tf(tensor_base)

    # 6. ImageNet Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm_tf = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_norm = norm_tf(tensor_base)

    # Convert CHW tensors to HWC numpy arrays for display
    def to_disp(t):
        arr = t.permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(arr, 0.0, 1.0)

    # 7. Diagnostic 6-panel visual comparison
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor="#0f172a")

    panels = [
        ((0, 0), to_disp(tensor_base), f"1. Base Image Tensor ({tensor_base.shape[1]}x{tensor_base.shape[2]})\nv2.ToImage + ToDtype(scale=True)", "#38bdf8"),
        ((0, 1), to_disp(tensor_resized), f"2. Downsampled ({tensor_resized.shape[1]}x{tensor_resized.shape[2]})\nv2.Resize((32, 32))", "#f8fafc"),
        ((0, 2), to_disp(tensor_rotated), "3. Rotated (+30°)\nv2.RandomRotation", "#34d399"),
        ((1, 0), to_disp(tensor_flipped), "4. Forced Horizontal Flip\nv2.RandomHorizontalFlip(p=1.0)", "#f43f5e"),
        ((1, 1), to_disp(tensor_jittered), "5. Color Jitter (Br/Ct/Sat)\nv2.ColorJitter", "#fde047"),
        ((1, 2), np.clip((tensor_norm.permute(1, 2, 0).numpy() + 2) / 4.0, 0, 1), "6. ImageNet Normalized\nv2.Normalize(mean, std)", "#a78bfa"),
    ]

    for (pos, img_data, title, col) in panels:
        ax = axes[pos[0], pos[1]]
        ax.imshow(img_data)
        ax.set_title(title, color=col, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.5)

    plt.suptitle("TorchVision Lab 02: Visualizing torchvision.transforms.v2 Pipelines",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "02-transforms-v2.png"
    preview_file = PREVIEWS / "02-transforms-v2.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 02 figure to {out_file} and {preview_file}")
    return {
        "original_shape": [orig_h, orig_w, 3],
        "tensor_base_shape": list(tensor_base.shape),
        "tensor_resized_shape": list(tensor_resized.shape),
        "transforms_applied": ["Resize", "RandomRotation", "RandomHorizontalFlip", "ColorJitter", "Normalize"],
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
