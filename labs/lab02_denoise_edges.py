"""Lab 02: Denoising, Quality Metrics (PSNR & SSIM) and Edge Detection with scikit-image."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature, filters, io, metrics, morphology, util
from labs.common import DATA, OUTPUTS, PREVIEWS


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    ref_path = DATA / "cells-reference.png"
    noisy_path = DATA / "cells-noisy.png"
    if not ref_path.exists() or not noisy_path.exists():
        from data.generate_samples import generate_all
        generate_all()

    # 1. Load reference & noisy images normalized to [0.0, 1.0]
    ref = util.img_as_float(io.imread(str(ref_path)))
    noisy = util.img_as_float(io.imread(str(noisy_path)))

    # 2. Denoising methods
    denoised_gauss = filters.gaussian(noisy, sigma=1.2)
    denoised_median = filters.median(noisy, footprint=morphology.disk(2))

    # 3. Objective Image Quality Assessment
    psnr_noisy = metrics.peak_signal_noise_ratio(ref, noisy, data_range=1.0)
    ssim_noisy = metrics.structural_similarity(ref, noisy, data_range=1.0)

    psnr_gauss = metrics.peak_signal_noise_ratio(ref, denoised_gauss, data_range=1.0)
    ssim_gauss = metrics.structural_similarity(ref, denoised_gauss, data_range=1.0)

    psnr_median = metrics.peak_signal_noise_ratio(ref, denoised_median, data_range=1.0)
    ssim_median = metrics.structural_similarity(ref, denoised_median, data_range=1.0)

    # 4. Feature Extraction: Sobel and Canny
    sobel = filters.sobel(denoised_gauss)
    canny = feature.canny(denoised_gauss, sigma=1.2)

    # 5. Diagnostic 6-panel visual comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="#0f172a")

    # (0, 0) Reference
    axes[0, 0].imshow(ref, cmap="gray")
    axes[0, 0].set_title("1. Ground Truth Reference\nPSNR: inf | SSIM: 1.000", color="#f8fafc", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    # (0, 1) Noisy Input
    axes[0, 1].imshow(noisy, cmap="gray")
    axes[0, 1].set_title(f"2. Noisy Input Image\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.3f}", color="#f8fafc", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    # (0, 2) Gaussian Filter
    axes[0, 2].imshow(denoised_gauss, cmap="gray")
    axes[0, 2].set_title(f"3. Gaussian Filter (sigma=1.2)\nPSNR: {psnr_gauss:.2f} dB | SSIM: {ssim_gauss:.3f}", color="#38bdf8", fontsize=10, fontweight="bold")
    axes[0, 2].axis("off")

    # (1, 0) Median Filter
    axes[1, 0].imshow(denoised_median, cmap="gray")
    axes[1, 0].set_title(f"4. Median Filter (disk r=2)\nPSNR: {psnr_median:.2f} dB | SSIM: {ssim_median:.3f}", color="#34d399", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")

    # (1, 1) Sobel Gradient
    axes[1, 1].imshow(sobel, cmap="viridis")
    axes[1, 1].set_title(f"5. Sobel Gradient Magnitude\nMax Gradient: {sobel.max():.3f}", color="#f8fafc", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")

    # (1, 2) Canny Edges
    axes[1, 2].imshow(canny, cmap="magma")
    axes[1, 2].set_title(f"6. Canny Binary Edge Map\nEdge Pixels: {int(canny.sum())}", color="#f43f5e", fontsize=10, fontweight="bold")
    axes[1, 2].axis("off")

    plt.suptitle("scikit-image Lab 02: Denoising, Quality Metrics & Edge Detection", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "02-denoise-edges-metrics.png"
    preview_file = PREVIEWS / "02-denoise-edges-metrics.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 02 figure to {out_file} and {preview_file}")
    return {
        "psnr_noisy": float(psnr_noisy),
        "ssim_noisy": float(ssim_noisy),
        "psnr_gauss": float(psnr_gauss),
        "ssim_gauss": float(ssim_gauss),
        "psnr_median": float(psnr_median),
        "ssim_median": float(ssim_median),
        "canny_edge_pixels": int(canny.sum()),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
