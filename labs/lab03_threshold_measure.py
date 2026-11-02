"""Lab 03: Otsu Thresholding, Morphology and Regionprops Measurement with scikit-image."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import filters, io, measure, morphology
from labs.common import DATA, OUTPUTS, PREVIEWS


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    input_path = DATA / "cells-synthetic.png"
    if not input_path.exists():
        from data.generate_samples import generate_all
        generate_all()

    # 1. Load grayscale synthetic cells image
    img = io.imread(str(input_path))

    # 2. Otsu Global Thresholding
    thresh = filters.threshold_otsu(img)
    raw_mask = img > thresh

    # 3. Morphology: Opening and Noise Removal (scikit-image 0.26 API)
    opened = morphology.opening(raw_mask, footprint=morphology.disk(2))
    cleaned = morphology.remove_small_objects(opened, max_size=50)
    cleaned = morphology.remove_small_holes(cleaned, max_size=50)

    # 4. Connected Component Labeling
    labels = measure.label(cleaned, connectivity=2)
    num_objects = int(labels.max())
    assert num_objects == 8, f"Expected exactly 8 synthetic cells, found {num_objects}"

    # 5. Region Property Extraction
    props = measure.regionprops_table(
        labels,
        properties=("label", "area", "perimeter", "eccentricity", "centroid")
    )
    df = pd.DataFrame(props)
    # Add calibrated physical units (assuming 0.5 um per pixel)
    df["area_um2"] = (df["area"] * 0.25).round(2)
    df["perimeter_um"] = (df["perimeter"] * 0.5).round(2)
    df["centroid_y"] = df["centroid-0"].round(2)
    df["centroid_x"] = df["centroid-1"].round(2)

    csv_file = OUTPUTS / "03-region-measurements.csv"
    df.to_csv(csv_file, index=False)

    # 6. Diagnostic 4-panel visual figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f172a")

    # (0, 0) Raw image
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title(f"1. Raw Image (Otsu T = {thresh:.1f})", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # (0, 1) Raw Otsu Mask
    axes[0, 1].imshow(raw_mask, cmap="gray")
    axes[0, 1].set_title(f"2. Raw Otsu Mask (Noise Speckles: {raw_mask.sum()} px)", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # (1, 0) Cleaned Morphology Mask
    axes[1, 0].imshow(cleaned, cmap="gray")
    axes[1, 0].set_title(f"3. Morphological Cleanup (max_size=50: {cleaned.sum()} px)", color="#38bdf8", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # (1, 1) Labeled Regions with Overlays
    colored_labels = np.ma.masked_where(labels == 0, labels)
    axes[1, 1].imshow(img, cmap="gray", alpha=0.6)
    im = axes[1, 1].imshow(colored_labels, cmap="tab10", alpha=0.7)
    for _, row in df.iterrows():
        cy = row["centroid-0"]
        cx = row["centroid-1"]
        lbl = int(row["label"])
        area = int(row["area"])
        axes[1, 1].plot(cx, cy, "r+", markersize=10, markeredgewidth=2)
        axes[1, 1].text(cx + 8, cy - 8, f"#{lbl} ({area}px)", color="#fde047", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.8, edgecolor="#334155"))

    axes[1, 1].set_title(f"4. Measured Components ({num_objects} Cells Identified)", color="#34d399", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    plt.suptitle("scikit-image Lab 03: Otsu Thresholding, Morphology & Regionprops", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "03-threshold-regionprops.png"
    preview_file = PREVIEWS / "03-threshold-regionprops.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 03 figure to {out_file} and CSV to {csv_file}")
    return {
        "num_objects": num_objects,
        "threshold_value": float(thresh),
        "total_foreground_pixels": int(cleaned.sum()),
        "csv_path": str(csv_file),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
