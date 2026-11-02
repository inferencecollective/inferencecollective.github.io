"""Lab 4: Batch format conversion, JPEG/PNG/WebP compression, and manifest audit."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    scenes = sorted(DATA_DIR.glob("scene-[0-9]*.png"))
    if len(scenes) < 4:
        from data.generate_samples import generate_all
        generate_all(DATA_DIR)
        scenes = sorted(DATA_DIR.glob("scene-[0-9]*.png"))

    records = []
    export_dir = output_dir / "batch_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    for path in scenes:
        stem = path.stem
        with Image.open(path) as src:
            rgb = ImageOps.exif_transpose(src).convert("RGB")
            w, h = rgb.size

            # 1. High-quality JPEG (Quality=90)
            jpg_90_path = export_dir / f"{stem}_q90.jpg"
            rgb.save(jpg_90_path, "JPEG", quality=90, optimize=True)

            # 2. Medium-quality JPEG (Quality=60)
            jpg_60_path = export_dir / f"{stem}_q60.jpg"
            rgb.save(jpg_60_path, "JPEG", quality=60, optimize=True)

            # 3. Lossless Optimized PNG
            png_path = export_dir / f"{stem}_opt.png"
            rgb.save(png_path, "PNG", optimize=True)

            # 4. WebP (Quality=80)
            webp_path = export_dir / f"{stem}_q80.webp"
            try:
                rgb.save(webp_path, "WEBP", quality=80)
                webp_size = webp_path.stat().st_size
            except Exception:
                webp_size = 0

            records.append({
                "scene": stem,
                "width": w,
                "height": h,
                "orig_png_kb": round(path.stat().st_size / 1024, 1),
                "opt_png_kb": round(png_path.stat().st_size / 1024, 1),
                "jpeg_90_kb": round(jpg_90_path.stat().st_size / 1024, 1),
                "jpeg_60_kb": round(jpg_60_path.stat().st_size / 1024, 1),
                "webp_80_kb": round(webp_size / 1024, 1),
            })

    # Write CSV Manifest
    csv_path = output_dir / "manifest_format_sizes.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Plot Multi-Format Comparison Chart
    df = pd.DataFrame(records)
    scenes_list = df["scene"].tolist()
    x = np.arange(len(scenes_list))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.bar(x - 1.5 * width, df["opt_png_kb"], width, label="Optimized PNG", color="#0284c7")
    ax.bar(x - 0.5 * width, df["jpeg_90_kb"], width, label="JPEG (Q=90)", color="#f59e0b")
    ax.bar(x + 0.5 * width, df["jpeg_60_kb"], width, label="JPEG (Q=60)", color="#ea580c")
    ax.bar(x + 1.5 * width, df["webp_80_kb"], width, label="WebP (Q=80)", color="#10b981")

    ax.set_ylabel("File Size (KB)", fontsize=11)
    ax.set_title("Pillow Multi-Format Image Export & Size Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes_list, fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    for target_dir in [output_dir, PREVIEWS_DIR]:
        fig.savefig(target_dir / "04-format-comparison.png", dpi=160)
    plt.close(fig)

    metrics = {
        "scenes_processed": len(scenes),
        "manifest_path": str(csv_path),
        "avg_png_kb": round(df["opt_png_kb"].mean(), 1),
        "avg_jpeg90_kb": round(df["jpeg_90_kb"].mean(), 1),
        "avg_webp80_kb": round(df["webp_80_kb"].mean(), 1),
    }
    print(f"Lab 4 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
