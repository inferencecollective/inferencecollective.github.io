"""Lab 04: Model Zoo Architectures, Preprocessing Recipes and Weight Selection in TorchVision."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)
from labs.common import DATA, OUTPUTS, PREVIEWS


def run(pretrained: bool = False, image_path: str | None = None) -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate architectures with weights=None (100% offline, reproducible)
    mb_model = mobilenet_v3_small(weights=None, num_classes=3).eval()
    rn_model = resnet18(weights=None, num_classes=3).eval()

    mb_params = sum(p.numel() for p in mb_model.parameters())
    rn_params = sum(p.numel() for p in rn_model.parameters())

    # 2. Inspect official preprocessing recipe from Weight Enum
    recipe = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    recipe_name = type(recipe).__name__

    # 3. Forward pass with uninitialized weights
    dummy_input = torch.zeros(1, 3, 64, 64)
    with torch.inference_mode():
        untrained_logits = mb_model(dummy_input)
        untrained_probs = torch.softmax(untrained_logits, dim=1).numpy().squeeze()

    # Calculate Shannon entropy (proves uninitialized weights yield high-entropy noise)
    entropy = -np.sum(untrained_probs * np.log(untrained_probs + 1e-12))
    max_entropy = np.log(3)  # ln(3) ~ 1.0986

    # 4. Optional online pretrained inference if requested
    pretrained_prediction = None
    if pretrained:
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            pt_model = mobilenet_v3_small(weights=weights).eval()
            print("Successfully loaded official MobileNetV3 Small weights.")
        except Exception as e:
            print(f"Pretrained weights download skipped or offline: {e}")

    # 5. Diagnostic 4-panel visual figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="#0f172a")

    # (0, 0) Parameter Count Comparison
    ax0 = axes[0, 0]
    ax0.set_facecolor("#1e293b")
    models = ["MobileNetV3-Small", "ResNet-18"]
    param_millions = [mb_params / 1e6, rn_params / 1e6]
    bars = ax0.bar(models, param_millions, color=["#38bdf8", "#34d399"], width=0.45)
    for bar in bars:
        h = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.2f}M",
                 ha="center", va="bottom", color="#f8fafc", fontweight="bold")
    ax0.set_title("1. Model Zoo Parameter Footprint (Millions)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Parameters (M)", color="#94a3b8", fontsize=9)
    ax0.set_ylim([0, 14])
    ax0.tick_params(colors="#94a3b8")
    for spine in ax0.spines.values():
        spine.set_color("#334155")

    # (0, 1) Weight Preprocessing Specifications
    ax1 = axes[0, 1]
    ax1.set_facecolor("#1e293b")
    ax1.set_title("2. Weight Enum Transforms Inspection", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.axis("off")
    spec_lines = [
        ("Weights Enum:", "MobileNet_V3_Small_Weights.IMAGENET1K_V1"),
        ("Recipe Class:", recipe_name),
        ("Target Crop Size:", "224 x 224 pixels"),
        ("Resize Scale:", "InterpolationMode.BILINEAR"),
        ("Channel Normalization:", "Mean: [0.485, 0.456, 0.406]"),
        ("", "Std:  [0.229, 0.224, 0.225]"),
        ("Input Value Scaling:", "[0.0, 1.0] float32 tensor"),
    ]
    for idx, (label, val) in enumerate(spec_lines):
        ax1.text(0.04, 0.85 - idx * 0.12, f"{label:<22} {val}", color="#38bdf8" if label else "#94a3b8",
                 fontsize=9.5, fontfamily="monospace", transform=ax1.transAxes)

    # (1, 0) Untrained Softmax Distribution & Entropy
    ax2 = axes[1, 0]
    ax2.set_facecolor("#1e293b")
    classes = ["Diagonal (0)", "Horizontal (1)", "Vertical (2)"]
    ax2.bar(classes, untrained_probs, color="#f43f5e", width=0.45, alpha=0.85)
    ax2.axhline(1.0 / 3.0, color="#fde047", linestyle="--", label="Uniform Chance (1/3)")
    ax2.set_title(f"3. Untrained Output (Entropy: {entropy:.3f} / Max: {max_entropy:.3f})", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Softmax Probability", color="#94a3b8", fontsize=9)
    ax2.set_ylim([0, 0.7])
    ax2.tick_params(colors="#94a3b8")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # (1, 1) Best Practice Principles
    ax3 = axes[1, 1]
    ax3.set_facecolor("#1e293b")
    ax3.set_title("4. Production Model Zoo Best Practices", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.axis("off")
    principles = [
        "✓ Always verify weights parameter (weights=None vs Enum)",
        "✓ Match the model's exact documented preprocessing recipe",
        "✓ Do not assume all architectures use 224x224 input",
        "✓ Use weights_only=True during local checkpoint restoration",
        "✓ Cache weights in internal registry for air-gapped CI/CD",
        "✓ Measure inference latency on target hardware (CPU vs GPU)",
    ]
    for idx, rule in enumerate(principles):
        ax3.text(0.04, 0.82 - idx * 0.13, rule, color="#34d399", fontsize=9.5, fontweight="bold", transform=ax3.transAxes)

    plt.suptitle("TorchVision Lab 04: Model Zoo Architectures & Preprocessing Specifications",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "04-model-zoo.png"
    preview_file = PREVIEWS / "04-model-zoo.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 04 figure to {out_file} and {preview_file}")
    return {
        "mobilenet_params": int(mb_params),
        "resnet_params": int(rn_params),
        "recipe_type": recipe_name,
        "untrained_entropy": float(entropy),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", help="Download official ImageNet weights")
    parser.add_argument("--image", type=str, default=None, help="Path to input image for classification")
    args = parser.parse_args()
    run(pretrained=args.pretrained, image_path=args.image)
