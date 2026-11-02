"""Lab 3: ImageDraw, Text bounding boxes, and RGBA Alpha Compositing Watermarks."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def apply_watermark(image: Image.Image, logo_path: Path, text: str = "AI Data Foundation • © 2026") -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    # 1. Overlay semi-transparent badge at bottom-right
    if logo_path.exists():
        with Image.open(logo_path) as logo_src:
            badge = logo_src.copy().convert("RGBA")
            badge.thumbnail((120, 120), Image.Resampling.LANCZOS)
            # Adjust opacity by scaling alpha channel
            r, g, b, a = badge.split()
            a = a.point(lambda p: int(p * 0.85))
            badge.putalpha(a)
            badge_pos = (base.width - badge.width - 25, base.height - badge.height - 25)
            overlay.alpha_composite(badge, badge_pos)

    # 2. Translucent Text Bar
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=18)
    banner_h = 44
    draw.rectangle([(0, base.height - banner_h), (base.width, base.height)], fill=(15, 23, 42, 175))

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text((25, base.height - banner_h + 12), text, font=font, fill=(248, 250, 252, 230))

    # Composite overlay onto base
    result = Image.alpha_composite(base, overlay)
    return result


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = DATA_DIR / "scene-03.png"
    logo_path = DATA_DIR / "logo.png"

    if not input_path.exists() or not logo_path.exists():
        from data.generate_samples import generate_all
        generate_all(DATA_DIR)

    with Image.open(input_path) as src:
        original = src.copy().convert("RGBA")

    watermarked = apply_watermark(original, logo_path)

    # Save standalone watermarked output
    watermarked.save(output_dir / "scene-03-watermarked.png")

    # Build Side-by-Side Comparison Canvas (1040 x 420)
    w, h = 500, 375
    left_img = ImageOps.fit(original, (w, h), Image.Resampling.LANCZOS)
    right_img = ImageOps.fit(watermarked, (w, h), Image.Resampling.LANCZOS)

    comp = Image.new("RGB", (1040, 440), "#0f172a")
    d = ImageDraw.Draw(comp)
    font = ImageFont.load_default(size=16)

    d.text((25, 12), "Original Image (Unmarked)", font=font, fill="#94a3b8")
    d.text((545, 12), "Watermarked (RGBA Alpha Compositing)", font=font, fill="#38bdf8")

    comp.paste(left_img.convert("RGB"), (20, 45))
    comp.paste(right_img.convert("RGB"), (540, 45))

    for target_dir in [output_dir, PREVIEWS_DIR]:
        comp.save(target_dir / "03-watermark-before-after.png")

    metrics = {
        "original_size": original.size,
        "watermarked_file": str(output_dir / "scene-03-watermarked.png"),
        "comparison_size": comp.size,
    }
    print(f"Lab 3 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
