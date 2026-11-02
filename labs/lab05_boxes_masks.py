"""Lab 05: TVTensors, Synchronized Bounding Boxes and Masks with transforms.v2."""
from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from labs.common import DATA, OUTPUTS, PREVIEWS


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    H, W = 48, 80

    # 1. Create synthetic annotated scene
    raw_img = torch.zeros(3, H, W, dtype=torch.uint8)
    raw_img[:, :, :] = torch.tensor([15, 23, 42], dtype=torch.uint8).view(3, 1, 1)  # dark background

    # Object 1: Horizontal bar (Cyan: [56, 189, 248]) at [6, 8, 32, 22]
    raw_img[:, 8:22, 6:32] = torch.tensor([56, 189, 248], dtype=torch.uint8).view(3, 1, 1)

    # Object 2: Vertical bar (Rose: [244, 63, 94]) at [45, 14, 65, 42]
    raw_img[:, 14:42, 45:65] = torch.tensor([244, 63, 94], dtype=torch.uint8).view(3, 1, 1)

    # Wrap in TVTensors
    img = tv_tensors.Image(raw_img)
    orig_box_coords = torch.tensor([
        [6.0, 8.0, 32.0, 22.0],
        [45.0, 14.0, 65.0, 42.0]
    ])
    boxes = tv_tensors.BoundingBoxes(orig_box_coords, format="XYXY", canvas_size=(H, W))

    raw_mask = torch.zeros(2, H, W, dtype=torch.uint8)
    raw_mask[0, 8:22, 6:32] = 1
    raw_mask[1, 14:42, 45:65] = 1
    mask = tv_tensors.Mask(raw_mask)

    # 2. Synchronized Forced Horizontal Flip (p=1.0)
    flip_tf = v2.RandomHorizontalFlip(p=1.0)
    flipped_img, flipped_boxes, flipped_mask = flip_tf(img, boxes, mask)

    # Verification: width W=80. For box 1 [6, 8, 32, 22]:
    # x1_new = 80 - 32 = 48, x2_new = 80 - 6 = 74
    expected_flipped_box0 = torch.tensor([48.0, 8.0, 74.0, 22.0])
    assert torch.allclose(flipped_boxes[0], expected_flipped_box0), f"Flipped box mismatch: {flipped_boxes[0]}"

    # 3. Synchronized Resizing to (72, 120)
    target_H, target_W = 72, 120
    resize_tf = v2.Resize((target_H, target_W))
    resized_img, resized_boxes, resized_mask = resize_tf(flipped_img, flipped_boxes, flipped_mask)

    # 4. Diagnostic 3-panel visual figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="#0f172a")

    panels_data = [
        (0, img, boxes, mask, "1. Original Annotated Scene\nCanvas: (48x80)", "#38bdf8"),
        (1, flipped_img, flipped_boxes, flipped_mask, "2. Synchronized Horizontal Flip\nv2.RandomHorizontalFlip(p=1.0)", "#f43f5e"),
        (2, resized_img, resized_boxes, resized_mask, f"3. Synchronized Resize\nv2.Resize(({target_H}, {target_W}))", "#34d399"),
    ]

    box_colors = ["#38bdf8", "#f43f5e"]
    for (idx, im_t, b_t, m_t, title, title_col) in panels_data:
        ax = axes[idx]
        ax.set_facecolor("#1e293b")
        im_np = im_t.permute(1, 2, 0).numpy()
        ax.imshow(im_np)

        # Plot bounding boxes and labels
        boxes_np = b_t.numpy()
        for b_idx in range(len(boxes_np)):
            x1, y1, x2, y2 = boxes_np[b_idx]
            w = x2 - x1
            h = y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=box_colors[b_idx], facecolor="none", linestyle="--")
            ax.add_patch(rect)
            ax.text(x1, max(0, y1 - 2), f"Obj {b_idx+1}: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]",
                    color=box_colors[b_idx], fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.8, edgecolor="#334155"))

        ax.set_title(title, color=title_col, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.5)

    plt.suptitle("TorchVision Lab 05: TVTensors Synchronized Bounding Boxes & Masks",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "05-boxes-masks-v2.png"
    preview_file = PREVIEWS / "05-boxes-masks-v2.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 05 figure to {out_file} and {preview_file}")
    return {
        "original_canvas": [H, W],
        "original_boxes": orig_box_coords.tolist(),
        "flipped_boxes": flipped_boxes.tolist(),
        "resized_boxes": resized_boxes.tolist(),
        "resized_canvas": [target_H, target_W],
        "alignment_verified": True,
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
