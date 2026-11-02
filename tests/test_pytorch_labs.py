"""Unit tests for PyTorch deep learning laboratories."""
from pathlib import Path
from labs import (
    lab01_tensors_autograd,
    lab02_mlp_classification,
    lab03_cnn_images,
    lab04_embeddings_nlp,
    lab05_tiny_transformer,
)


def test_lab01_tensors_autograd():
    res = lab01_tensors_autograd.run()
    assert abs(res["final_w"] - res["true_w"]) < 0.5
    assert abs(res["final_b"] - res["true_b"]) < 0.5
    assert res["final_loss"] < 1.5
    assert Path("outputs/01-tensors-autograd.png").exists()
    assert Path("previews/01-tensors-autograd.png").exists()


def test_lab02_mlp_classification():
    res = lab02_mlp_classification.run()
    assert res["final_test_acc"] > 0.85
    assert res["final_test_loss"] < 0.6
    assert Path("outputs/02-mlp-classification.png").exists()
    assert Path("previews/02-mlp-classification.png").exists()


def test_lab03_cnn_images():
    res = lab03_cnn_images.run()
    assert res["final_test_acc"] > 0.85
    assert res["num_parameters"] > 0
    assert Path("outputs/03-cnn-image-classification.png").exists()
    assert Path("previews/03-cnn-image-classification.png").exists()


def test_lab04_embeddings_nlp():
    res = lab04_embeddings_nlp.run()
    assert res["final_test_acc"] > 0.85
    assert res["vocab_size"] == 16
    assert Path("outputs/04-embeddings-nlp.png").exists()
    assert Path("previews/04-embeddings-nlp.png").exists()


def test_lab05_tiny_transformer():
    res = lab05_tiny_transformer.run()
    assert res["weights_verified"] is True
    assert res["total_parameters"] > 0
    assert Path(res["checkpoint_path"]).exists()
    assert Path("outputs/05-causal-attention.png").exists()
    assert Path("previews/05-causal-attention.png").exists()
