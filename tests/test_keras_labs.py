"""Unit tests for Keras 3 multi-backend deep learning laboratories."""
from pathlib import Path
from labs import (
    lab01_tensors_regression,
    lab02_classification_early_stopping,
    lab03_keras_cnn,
    lab04_embeddings_attention,
    lab05_backend_portability,
)


def test_lab01_tensors_regression():
    res = lab01_tensors_regression.run()
    assert res["test_rmse"] < 1.0
    assert Path("artifacts/01-tensors-regression.json").exists()
    assert Path("artifacts/01-tensors-regression.png").exists()
    assert Path("outputs/01-tensors-regression.png").exists()
    assert Path("previews/01-tensors-regression.png").exists()


def test_lab02_classification_early_stopping():
    res = lab02_classification_early_stopping.run()
    assert res["test_accuracy"] > 0.85
    assert Path("artifacts/02-classification-early-stopping.json").exists()
    assert Path("artifacts/02-classification-early-stopping.png").exists()
    assert Path("outputs/02-classification-early-stopping.png").exists()
    assert Path("previews/02-classification-early-stopping.png").exists()


def test_lab03_keras_cnn():
    res = lab03_keras_cnn.run()
    assert res["test_accuracy"] > 0.85
    assert Path("artifacts/03-cnn-image-classifier.json").exists()
    assert Path("artifacts/03-cnn-image-classifier.png").exists()
    assert Path("outputs/03-cnn-image-classifier.png").exists()
    assert Path("previews/03-cnn-image-classifier.png").exists()


def test_lab04_embeddings_attention():
    res = lab04_embeddings_attention.run()
    assert res["test_accuracy"] > 0.85
    assert Path("artifacts/04-embeddings-attention.json").exists()
    assert Path("artifacts/04-embeddings-attention.png").exists()
    assert Path("outputs/04-embeddings-attention.png").exists()
    assert Path("previews/04-embeddings-attention.png").exists()


def test_lab05_backend_portability():
    res = lab05_backend_portability.run()
    assert res["parity_verified"] is True
    assert res["max_difference"] < 1e-4
    assert Path("artifacts/05-backend-portability.json").exists()
    assert Path("artifacts/05-backend-portability.png").exists()
    assert Path("artifacts/portable_model.keras").exists()
    assert Path("outputs/05-backend-portability.png").exists()
    assert Path("previews/05-backend-portability.png").exists()
