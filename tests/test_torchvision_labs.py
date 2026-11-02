"""Unit tests for TorchVision computer vision laboratories."""
from pathlib import Path
from labs import (
    lab01_imagefolder_grid,
    lab02_transforms_v2,
    lab03_train_classifier,
    lab04_model_zoo,
    lab05_boxes_masks,
)


def test_lab01_imagefolder_grid():
    res = lab01_imagefolder_grid.run()
    assert res["classes"] == ["diagonal", "horizontal", "vertical"]
    assert res["batch_shape"] == [12, 3, 64, 64]
    assert Path("outputs/01-imagefolder-grid.png").exists()
    assert Path("previews/01-imagefolder-grid.png").exists()


def test_lab02_transforms_v2():
    res = lab02_transforms_v2.run()
    assert res["original_shape"] == [64, 64, 3]
    assert "RandomHorizontalFlip" in res["transforms_applied"]
    assert Path("outputs/02-transforms-v2.png").exists()
    assert Path("previews/02-transforms-v2.png").exists()


def test_lab03_train_classifier():
    res = lab03_train_classifier.run()
    assert res["test_acc"] > 0.85
    assert Path(res["metadata_path"]).exists()
    assert Path(res["checkpoint_path"]).exists()
    assert Path("outputs/03-classifier-training.png").exists()
    assert Path("previews/03-classifier-training.png").exists()


def test_lab04_model_zoo():
    res = lab04_model_zoo.run()
    assert res["mobilenet_params"] > 1_000_000
    assert res["resnet_params"] > 10_000_000
    assert Path("outputs/04-model-zoo.png").exists()
    assert Path("previews/04-model-zoo.png").exists()


def test_lab05_boxes_masks():
    res = lab05_boxes_masks.run()
    assert res["alignment_verified"] is True
    assert len(res["flipped_boxes"]) == 2
    assert Path("outputs/05-boxes-masks-v2.png").exists()
    assert Path("previews/05-boxes-masks-v2.png").exists()
