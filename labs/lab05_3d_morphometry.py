"""Lab 05: 3D Volume Segmentation and Morphometry with scikit-image."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import filters, measure
from labs.common import DATA, OUTPUTS, PREVIEWS


def run() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    input_path = DATA / "volume-cells.npy"
    if not input_path.exists():
        from data.generate_samples import generate_all
        generate_all()

    # 1. Load 3D volumetric stack (Z, Y, X)
    volume = np.load(str(input_path))
    z_depth, y_height, x_width = volume.shape

    # 2. 3D Global Otsu thresholding
    thresh = filters.threshold_otsu(volume)
    mask = volume > thresh

    # 3. 3D Connected component labeling (connectivity=3 gives full 26-connectivity)
    labels = measure.label(mask, connectivity=3)
    num_regions = int(labels.max())
    assert num_regions == 3, f"Expected 3 synthetic 3D spherical bodies, found {num_regions}"

    # 4. Volumetric morphometry with regionprops_table
    props = measure.regionprops_table(
        labels,
        properties=("label", "area", "centroid", "bbox")
    )
    df = pd.DataFrame(props)

    # Physical spacing calibration: (dz=1.5 um, dy=0.5 um, dx=0.5 um)
    voxel_volume_um3 = 1.5 * 0.5 * 0.5
    df["volume_voxels"] = df["area"]
    df["volume_um3"] = (df["area"] * voxel_volume_um3).round(2)
    df["centroid_z"] = df["centroid-0"].round(2)
    df["centroid_y"] = df["centroid-1"].round(2)
    df["centroid_x"] = df["centroid-2"].round(2)

    csv_file = OUTPUTS / "05-3d-region-measurements.csv"
    df.to_csv(csv_file, index=False)

    # 5. Diagnostic 4-panel visual figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="#0f172a")

    z_mid = z_depth // 2

    # (0, 0) Middle Z-Slice Raw
    axes[0, 0].imshow(volume[z_mid, :, :], cmap="gray")
    axes[0, 0].set_title(f"1. Middle Slice (Z={z_mid}/{z_depth}) - Raw Intensity", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # (0, 1) Middle Z-Slice Segmentation Mask
    slice_lbl = labels[z_mid, :, :]
    masked_slice = np.ma.masked_where(slice_lbl == 0, slice_lbl)
    axes[0, 1].imshow(volume[z_mid, :, :], cmap="gray", alpha=0.5)
    axes[0, 1].imshow(masked_slice, cmap="tab10", alpha=0.85)
    axes[0, 1].set_title(f"2. Middle Slice (Z={z_mid}) - Labeled Cross-Sections", color="#38bdf8", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # (1, 0) Maximum Intensity Projection (MIP) along Z
    mip = np.max(volume, axis=0)
    axes[1, 0].imshow(mip, cmap="magma")
    axes[1, 0].set_title("3. Maximum Intensity Projection (Z-MIP)", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # (1, 1) 3D Labeled Projection & Projected Centroids
    label_mip = np.max(labels, axis=0)
    masked_mip = np.ma.masked_where(label_mip == 0, label_mip)
    axes[1, 1].imshow(mip, cmap="gray", alpha=0.45)
    axes[1, 1].imshow(masked_mip, cmap="tab10", alpha=0.75)
    for _, row in df.iterrows():
        cx = row["centroid-2"]
        cy = row["centroid-1"]
        cz = row["centroid-0"]
        lbl = int(row["label"])
        vol_vox = int(row["volume_voxels"])
        vol_um3 = row["volume_um3"]
        axes[1, 1].plot(cx, cy, "r+", markersize=12, markeredgewidth=2)
        axes[1, 1].text(cx + 4, cy - 4, f"#{lbl} (Z={cz:.1f})\n{vol_vox} vx / {vol_um3} μm³",
                        color="#fde047", fontsize=8.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.85, edgecolor="#334155"))

    axes[1, 1].set_title(f"4. 3D Projected Morphometry ({num_regions} Objects)", color="#34d399", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    plt.suptitle("scikit-image Lab 05: 3D Volumetric Segmentation & Morphometry", color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = OUTPUTS / "05-3d-volume-morphometry.png"
    preview_file = PREVIEWS / "05-3d-volume-morphometry.png"
    plt.savefig(str(out_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(preview_file), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Exported Lab 05 figure to {out_file} and CSV to {csv_file}")
    return {
        "volume_shape": volume.shape,
        "num_regions": num_regions,
        "voxel_counts": df["volume_voxels"].tolist(),
        "csv_path": str(csv_file),
        "output_path": str(out_file),
    }


if __name__ == "__main__":
    run()
