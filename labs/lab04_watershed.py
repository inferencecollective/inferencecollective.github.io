"""Lab 04: Watershed Segmentation for Separating Touching Objects with scikit-image."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, io, measure, segmentation
from labs.common import DATA, OUTPUTS, PREVIEWS


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    input_path = DATA / "cells-touching.png"
    if not input_path.exists():
        from data.generate_samples import generate_all
        generate_all()

    # 1. Load touching cells image
    img = io.imread(str(input_path))

    # 2. Otsu threshold to obtain foreground mask
    thresh = filters.threshold_otsu(img)
    mask = img > thresh

    # 3. Before watershed: standard connected components merges touching bodies
    naive_labels = measure.label(mask, connectivity=2)
    num_naive = int(naive_labels.max())

    # 4. Euclidean Distance Transform (EDT)
    distance = ndi.distance_transform_edt(mask)

    # 5. Peak local maxima detection for markers
    coords = feature.peak_local_max(distance, min_distance=25, labels=mask)
    num_peaks = len(coords)
    assert num_peaks == 8, f"Expected 8 local peak seeds, detected {num_peaks}"

    # 6. Build marker array
    markers = np.zeros(distance.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, 1):
        markers[r, c] = idx

    # 7. Watershed segmentation on inverted distance landscape
    ws_labels = segmentation.watershed(-distance, markers, mask=mask)
    num_cells = len(np.unique(ws_labels)) - 1
    assert num_cells == 8, f"Expected 8 separated cells, obtained {num_cells}"

    # 8. Boundary extraction
    boundaries = segmentation.find_boundaries(ws_labels, mode="thick")

    # 9. Diagnostic 4-panel visual figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f172a")

    # (0, 0) Original image
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("1. Original Image: Touching Cell Pairs", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # (0, 1) Naive Binary Mask (Merged)
    axes[0, 1].imshow(naive_labels, cmap="Set2")
    axes[0, 1].set_title(f"2. Naive Connected Components: Only {num_naive} Merged Objects", color="#f43f5e", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # (1, 0) Distance Transform + Seeds
    axes[1, 0].imshow(distance, cmap="viridis")
    axes[1, 0].plot(coords[:, 1], coords[:, 0], "r*", markersize=10, markeredgewidth=1.5, label=f"Seeds (n={num_peaks})")
    axes[1, 0].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")
    axes[1, 0].set_title("3. Euclidean Distance Transform + Seeds", color="#38bdf8", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # (1, 1) Final Watershed Labeled Segments + Boundaries
    display_img = np.stack([img, img, img], axis=-1)
    display_img[boundaries] = [244, 63, 94]  # Highlight separating boundaries in rose/red
    axes[1, 1].imshow(display_img)
    # Overlay colored labels with transparency
    masked_ws = np.ma.masked_where(ws_labels == 0, ws_labels)
    axes[1, 1].imshow(masked_ws, cmap="tab10", alpha=0.55)
    for idx, (r, c) in enumerate(coords, 1):
        axes[1, 1].text(c + 4, r - 4, f"#{idx}", color="#fde047", fontsize=9, fontweight="bold")

    axes[1, 1].set_title(f"4. Watershed Segmentation: {num_cells} Disentangled Cells", color="#34d399", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    plt.suptitle("scikit-image Lab 04: Watershed Segmentation for Touching Objects", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "04-watershed-touching-cells.png"
    preview_file = PREVIEWS / "04-watershed-touching-cells.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 04 figure to {out_file} and {preview_file}")
    return {
        "naive_clusters": num_naive,
        "peak_seeds": num_peaks,
        "segmented_cells": num_cells,
        "boundary_pixels": int(boundaries.sum()),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
