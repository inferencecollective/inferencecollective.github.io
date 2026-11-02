"""Unit tests for scikit-image scientific image processing labs."""
from pathlib import Path
import pandas as pd
from labs import (
    lab01_image_basics,
    lab02_denoise_edges,
    lab03_threshold_measure,
    lab04_watershed,
    lab05_3d_morphometry,
)


def test_lab01_image_basics():
    res = lab01_image_basics.run()
    assert res["original_shape"] == (400, 640, 3)
    assert 0.0 <= res["gray_min"] <= res["gray_max"] <= 1.0
    assert Path("outputs/01-images-and-contrast.png").exists()
    assert Path("previews/01-images-and-contrast.png").exists()


def test_lab02_denoise_edges():
    res = lab02_denoise_edges.run()
    assert res["psnr_gauss"] > res["psnr_noisy"]
    assert res["ssim_gauss"] > res["ssim_noisy"]
    assert res["canny_edge_pixels"] > 0
    assert Path("outputs/02-denoise-edges-metrics.png").exists()
    assert Path("previews/02-denoise-edges-metrics.png").exists()


def test_lab03_threshold_measure():
    res = lab03_threshold_measure.run()
    assert res["num_objects"] == 8
    csv_file = Path("outputs/03-region-measurements.csv")
    assert csv_file.exists()
    df = pd.read_csv(csv_file)
    assert len(df) == 8
    assert set(df["label"]) == set(range(1, 9))
    assert Path("outputs/03-threshold-regionprops.png").exists()


def test_lab04_watershed():
    res = lab04_watershed.run()
    assert res["naive_clusters"] == 4
    assert res["peak_seeds"] == 8
    assert res["segmented_cells"] == 8
    assert Path("outputs/04-watershed-touching-cells.png").exists()
    assert Path("previews/04-watershed-touching-cells.png").exists()


def test_lab05_3d_morphometry():
    res = lab05_3d_morphometry.run()
    assert res["num_regions"] == 3
    assert len(res["voxel_counts"]) == 3
    csv_file = Path("outputs/05-3d-region-measurements.csv")
    assert csv_file.exists()
    df = pd.read_csv(csv_file)
    assert len(df) == 3
    assert Path("outputs/05-3d-volume-morphometry.png").exists()
    assert Path("previews/05-3d-volume-morphometry.png").exists()
