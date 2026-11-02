"""Lab 01: RGB, Grayscale and Contrast Equalization with scikit-image."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, exposure, io
from labs.common import DATA, OUTPUTS, PREVIEWS


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    input_path = DATA / "scene-rgb.png"
    if not input_path.exists():
        from data.generate_samples import generate_all
        generate_all()

    # 1. Read RGB image
    rgb = io.imread(str(input_path))
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"Expected 3-channel RGB image, got {rgb.shape}"

    # 2. Convert to grayscale
    gray = color.rgb2gray(rgb)  # returns float64 in [0.0, 1.0]

    # 3. Apply Adaptive Histogram Equalization (CLAHE)
    clahe = exposure.equalize_adapthist(gray, clip_limit=0.03)

    # 4. Global histogram equalization for comparison
    global_eq = exposure.equalize_hist(gray)

    # 5. Diagnostic 4-panel visual plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="#0f172a")

    # Panel 0,0: Original RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"1. Original RGB: {rgb.shape} [{rgb.dtype}]", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 0,1: Grayscale (float [0, 1])
    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title(f"2. Grayscale: min={gray.min():.2f}, max={gray.max():.2f}", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel 1,0: Contrast Limited Adaptive Hist Eq (CLAHE)
    axes[1, 0].imshow(clahe, cmap="magma")
    axes[1, 0].set_title("3. CLAHE Enhanced (cmap='magma')", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Panel 1,1: Pixel Intensity Histograms
    ax_hist = axes[1, 1]
    ax_hist.set_facecolor("#1e293b")
    ax_hist.hist(gray.ravel(), bins=64, range=(0, 1), density=True, color="#94a3b8", alpha=0.6, label="Original Gray")
    ax_hist.hist(clahe.ravel(), bins=64, range=(0, 1), density=True, color="#38bdf8", alpha=0.6, label="CLAHE")
    ax_hist.set_title("4. Intensity Distribution Histograms", color="#f8fafc", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Normalized Intensity [0.0, 1.0]", color="#94a3b8", fontsize=9)
    ax_hist.set_ylabel("Pixel Density", color="#94a3b8", fontsize=9)
    ax_hist.tick_params(colors="#94a3b8")
    for spine in ax_hist.spines.values():
        spine.set_color("#334155")
    ax_hist.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper right")

    plt.suptitle("scikit-image Lab 01: RGB, Grayscale & Contrast Analysis", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "01-images-and-contrast.png"
    preview_file = PREVIEWS / "01-images-and-contrast.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 01 figure to {out_file} and {preview_file}")
    return {
        "original_shape": rgb.shape,
        "original_dtype": str(rgb.dtype),
        "gray_min": float(gray.min()),
        "gray_max": float(gray.max()),
        "clahe_mean": float(clahe.mean()),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
