"""Lab 1: Resize, Crop, and Thumbnail without stretching."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def add_label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=14)
    # Draw dark translucent banner at bottom
    w, h = out.size
    draw.rectangle([(0, h - 30), (w, h)], fill=(15, 23, 42, 220))
    draw.text((10, h - 22), text, font=font, fill="#f8fafc")
    return out


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = DATA_DIR / "scene-01.png"
    if not input_path.exists():
        from data.generate_samples import generate_all
        generate_all(DATA_DIR)

    with Image.open(input_path) as src:
        original = src.copy().convert("RGB")

    target_size = (360, 360)

    # 1. Non-proportional Direct Resize (stretches)
    stretched = original.resize(target_size, Image.Resampling.LANCZOS)
    stretched_lbl = add_label(stretched, "1. resize(): Exact size, Distorted Aspect Ratio")

    # 2. ImageOps.contain (Fits inside box, keeps ratio)
    contained = ImageOps.contain(original, target_size, Image.Resampling.LANCZOS)
    # Paste onto square canvas to standardize visual comparison
    contain_canvas = Image.new("RGB", target_size, "#090d16")
    offset_x = (target_size[0] - contained.size[0]) // 2
    offset_y = (target_size[1] - contained.size[1]) // 2
    contain_canvas.paste(contained, (offset_x, offset_y))
    contain_lbl = add_label(contain_canvas, "2. contain(): Proportional, Letterbox Gap")

    # 3. ImageOps.fit (Crops excess, exact square)
    fitted = ImageOps.fit(original, target_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    fitted_lbl = add_label(fitted, "3. fit(): Exact 360x360, Center Cropped")

    # 4. ImageOps.pad (Pads with custom background color)
    padded = ImageOps.pad(original, target_size, method=Image.Resampling.LANCZOS, color="#1e293b", centering=(0.5, 0.5))
    padded_lbl = add_label(padded, "4. pad(): Exact 360x360, Solid Border Padded")

    # Assemble 2x2 comparison grid (800x800)
    grid = Image.new("RGB", (760, 760), "#0f172a")
    grid.paste(stretched_lbl, (15, 15))
    grid.paste(contain_lbl, (385, 15))
    grid.paste(fitted_lbl, (15, 385))
    grid.paste(padded_lbl, (385, 385))

    for target_dir in [output_dir, PREVIEWS_DIR]:
        grid.save(target_dir / "01-resize-comparison.png")

    metrics = {
        "original_size": original.size,
        "target_size": target_size,
        "grid_size": grid.size,
    }
    print(f"Lab 1 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
