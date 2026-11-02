"""Lab 5: Secure Image Validation Pipeline and 2x2 Portfolio Contact Sheet."""
import json
import warnings
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MAX_FILE_SIZE = 20_000_000  # 20 MB
MAX_PIXELS = 24_000_000     # 24 Megapixels
ALLOWED_FORMATS = ["PNG", "JPEG", "WEBP"]


def validate_and_load(path: Path) -> Image.Image:
    """Defensive image verification pipeline before pixel decoding."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # 1. File Size Guard
    size_bytes = path.stat().st_size
    if size_bytes > MAX_FILE_SIZE:
        raise ValueError(f"File size {size_bytes} exceeds limit of {MAX_FILE_SIZE} bytes")

    # 2. Decompression Bomb & Format Probe
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)

        with Image.open(path, formats=ALLOWED_FORMATS) as probe:
            # 3. Dimension & Pixel Count Guard
            if probe.width * probe.height > MAX_PIXELS:
                raise ValueError(f"Image dimensions {probe.width}x{probe.height} exceed {MAX_PIXELS} pixels")

            # 4. Header Verification without pixel decode
            probe.verify()

    # 5. Clean Reopen & Pixel Load
    with Image.open(path, formats=ALLOWED_FORMATS) as src:
        img = ImageOps.exif_transpose(src).convert("RGB")
        img.load()
        return img


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    scenes = sorted(DATA_DIR.glob("scene-[0-9]*.png"))
    if len(scenes) < 4:
        from data.generate_samples import generate_all
        generate_all(DATA_DIR)
        scenes = sorted(DATA_DIR.glob("scene-[0-9]*.png"))

    validated_images = []
    audit_manifest = []

    for path in scenes[:4]:
        img = validate_and_load(path)
        validated_images.append(img)
        audit_manifest.append({
            "filename": path.name,
            "bytes": path.stat().st_size,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": "PNG",
            "status": "VALIDATED_SAFE",
        })

    # Build 2x2 Contact Sheet
    cell_w, cell_h = 420, 315
    pad = 20
    header_h = 60
    sheet_w = cell_w * 2 + pad * 3
    sheet_h = header_h + cell_h * 2 + pad * 3

    sheet = Image.new("RGB", (sheet_w, sheet_h), "#0f172a")
    draw = ImageDraw.Draw(sheet)

    title_font = ImageFont.load_default(size=20)
    label_font = ImageFont.load_default(size=13)

    # Sheet Header
    draw.text((pad, 20), "PILLOW DEFENSIVE AUDIT — 2×2 CONTACT SHEET", font=title_font, fill="#38bdf8")

    positions = [
        (pad, header_h + pad),
        (pad * 2 + cell_w, header_h + pad),
        (pad, header_h + pad * 2 + cell_h),
        (pad * 2 + cell_w, header_h + pad * 2 + cell_h),
    ]

    for idx, (img, pos, meta) in enumerate(zip(validated_images, positions, audit_manifest)):
        thumb = ImageOps.fit(img, (cell_w, cell_h), Image.Resampling.LANCZOS)
        # Add badge info bar
        t_draw = ImageDraw.Draw(thumb)
        t_draw.rectangle([(0, cell_h - 28), (cell_w, cell_h)], fill=(15, 23, 42, 220))
        tag = f"#{idx+1}: {meta['filename']} ({meta['width']}x{meta['height']} • {meta['bytes']//1024} KB)"
        t_draw.text((10, cell_h - 20), tag, font=label_font, fill="#f8fafc")

        # Paste onto sheet
        sheet.paste(thumb, pos)
        # Draw border
        draw.rectangle([pos, (pos[0] + cell_w, pos[1] + cell_h)], outline="#334155", width=2)

    for target_dir in [output_dir, PREVIEWS_DIR]:
        sheet.save(target_dir / "05-contact-sheet.png")

    manifest_path = output_dir / "contact_sheet_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(audit_manifest, f, indent=2)

    metrics = {
        "images_validated": len(validated_images),
        "contact_sheet_size": sheet.size,
        "manifest_path": str(manifest_path),
    }
    print(f"Lab 5 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
