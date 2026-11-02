"""Unit tests for Pillow image processing labs."""
from pathlib import Path
from PIL import Image
from labs import (
    lab01_resize_crop,
    lab02_enhance_filters,
    lab03_watermark,
    lab04_format_export,
    lab05_contact_sheet,
)


def test_lab01_resize():
    res = lab01_resize_crop.run()
    assert res["grid_size"] == (760, 760)
    out_file = Path("outputs/01-resize-comparison.png")
    assert out_file.exists()
    with Image.open(out_file) as im:
        assert im.size == (760, 760)


def test_lab02_enhance():
    res = lab02_enhance_filters.run()
    assert "operations" in res
    assert len(res["operations"]) == 6
    out_file = Path("outputs/02-enhancement-grid.png")
    assert out_file.exists()


def test_lab03_watermark():
    res = lab03_watermark.run()
    assert Path(res["watermarked_file"]).exists()
    out_file = Path("outputs/03-watermark-before-after.png")
    assert out_file.exists()


def test_lab04_format():
    res = lab04_format_export.run()
    assert res["scenes_processed"] == 4
    assert Path(res["manifest_path"]).exists()
    out_file = Path("outputs/04-format-comparison.png")
    assert out_file.exists()


def test_lab05_contact_sheet():
    res = lab05_contact_sheet.run()
    assert res["images_validated"] == 4
    assert Path(res["manifest_path"]).exists()
    out_file = Path("outputs/05-contact-sheet.png")
    assert out_file.exists()
