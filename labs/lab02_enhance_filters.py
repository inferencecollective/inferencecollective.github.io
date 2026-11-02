"""Lab 2: Image enhancements, spatial filters, blur, sharpen and contrast."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def add_tile_header(img: Image.Image, title: str) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + 34), "#0f172a")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    draw.text((10, 8), title, font=font, fill="#38bdf8")
    canvas.paste(img, (0, 34))
    return canvas


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = DATA_DIR / "scene-02.png"
    if not input_path.exists():
        from data.generate_samples import generate_all
        generate_all(DATA_DIR)

    with Image.open(input_path) as src:
        base = src.copy().convert("RGB")

    tile_size = (360, 270)
    thumb = ImageOps.fit(base, tile_size, Image.Resampling.LANCZOS)

    # 1. Original Base
    p1 = add_tile_header(thumb, "1. Original Base (360x270)")

    # 2. Brightness Enhance (1.35x)
    bright = ImageEnhance.Brightness(thumb).enhance(1.35)
    p2 = add_tile_header(bright, "2. Brightness (+35%)")

    # 3. Gaussian Blur (radius=4)
    blurred = thumb.filter(ImageFilter.GaussianBlur(radius=4))
    p3 = add_tile_header(blurred, "3. GaussianBlur (radius=4)")

    # 4. Unsharp Mask Sharpening
    sharp = thumb.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=2))
    p4 = add_tile_header(sharp, "4. UnsharpMask (radius=2, 160%)")

    # 5. AutoContrast
    autocontrast = ImageOps.autocontrast(thumb, cutoff=2)
    p5 = add_tile_header(autocontrast, "5. ImageOps.autocontrast()")

    # 6. Edge Detection Filter
    edges = thumb.filter(ImageFilter.FIND_EDGES)
    p6 = add_tile_header(edges, "6. ImageFilter.FIND_EDGES")

    # Assemble 2x3 Grid (W = 360*3 + padding = 1140, H = 304*2 + padding = 650)
    grid = Image.new("RGB", (1140, 650), "#090d16")
    grid.paste(p1, (15, 15))
    grid.paste(p2, (390, 15))
    grid.paste(p3, (765, 15))
    grid.paste(p4, (15, 335))
    grid.paste(p5, (390, 335))
    grid.paste(p6, (765, 335))

    for target_dir in [output_dir, PREVIEWS_DIR]:
        grid.save(target_dir / "02-enhancement-grid.png")

    metrics = {
        "tile_size": tile_size,
        "grid_size": grid.size,
        "operations": ["original", "brightness", "blur", "unsharp_mask", "autocontrast", "find_edges"],
    }
    print(f"Lab 2 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
