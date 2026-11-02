"""Unit tests for OpenCV computer vision labs."""
from pathlib import Path
import cv2
import numpy as np
from labs import (
    lab01_image_colors,
    lab02_filters_edges,
    lab03_contours,
    lab04_perspective,
    lab05_video_frames,
)


def test_lab01_colors():
    res = lab01_image_colors.run()
    assert res["original_shape"] == (400, 640, 3)
    assert res["orange_pixels"] > 0
    assert Path("outputs/01-color-and-resize.png").exists()


def test_lab02_filters():
    res = lab02_filters_edges.run()
    assert res["canny_edges"] > 0
    assert Path("outputs/02-filters-and-edges.png").exists()


def test_lab03_contours():
    res = lab03_contours.run()
    assert res["filtered_shapes"] == 5
    assert Path("outputs/03-contours-and-morphology.png").exists()


def test_lab04_perspective():
    res = lab04_perspective.process()
    assert res["aligned"].shape == (330, 480, 3)
    assert np.array(res["homography"]).shape == (3, 3)
    assert Path("outputs/04-document-perspective.png").exists()


def test_lab05_video():
    res = lab05_video_frames.process()
    assert res["frames_tracked"] == 12
    assert res["trajectory_points"] == 12
    assert Path("outputs/05-video-frame-tracking.png").exists()
