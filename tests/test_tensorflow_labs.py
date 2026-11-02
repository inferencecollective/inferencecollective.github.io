"""Unit tests for TensorFlow deep learning and deployment laboratories."""
from pathlib import Path
from labs import (
    lab01_tf_autograd,
    lab02_keras_classification,
    lab03_tfdata_cnn,
    lab04_savedmodel_serving,
    lab05_litert_export,
)


def test_lab01_tf_autograd():
    res = lab01_tf_autograd.run()
    assert abs(res["final_w"] - res["true_w"]) < 0.5
    assert abs(res["final_b"] - res["true_b"]) < 0.5
    assert res["final_loss"] < 0.5
    assert Path("artifacts/01-autograd-regression.json").exists()
    assert Path("artifacts/01-autograd-regression.png").exists()
    assert Path("outputs/01-autograd-regression.png").exists()
    assert Path("previews/01-autograd-regression.png").exists()


def test_lab02_keras_classification():
    res = lab02_keras_classification.run()
    assert res["test_accuracy"] > 0.85
    assert res["test_loss"] < 0.5
    assert Path("artifacts/02-keras-classification.json").exists()
    assert Path("artifacts/02-keras-classification.png").exists()
    assert Path("outputs/02-keras-classification.png").exists()
    assert Path("previews/02-keras-classification.png").exists()


def test_lab03_tfdata_cnn():
    res = lab03_tfdata_cnn.run()
    assert res["test_accuracy"] > 0.85
    assert res["pipeline_batch_size"] == 32
    assert res["input_shape"] == [28, 28, 1]
    assert Path("artifacts/03-tfdata-cnn.json").exists()
    assert Path("artifacts/03-tfdata-cnn.png").exists()
    assert Path("outputs/03-tfdata-cnn.png").exists()
    assert Path("previews/03-tfdata-cnn.png").exists()


def test_lab04_savedmodel_serving():
    res = lab04_savedmodel_serving.run()
    assert res["prediction_parity"] is True
    assert res["instances_tested"] == 2
    assert Path("artifacts/04-savedmodel-serving.json").exists()
    assert Path("artifacts/04-savedmodel-serving.png").exists()
    assert Path("artifacts/04-serving-request.json").exists()
    assert Path("artifacts/serving/1/saved_model.pb").exists()
    assert Path("outputs/04-savedmodel-serving.png").exists()
    assert Path("previews/04-savedmodel-serving.png").exists()


def test_lab05_litert_export():
    res = lab05_litert_export.run()
    assert res["compression_ratio"] > 2.0
    assert res["size_reduction_pct"] > 50.0
    assert res["parity_verified"] is True
    assert Path("artifacts/05-litert-export.json").exists()
    assert Path("artifacts/05-litert-export.png").exists()
    assert Path("artifacts/model.tflite").exists()
    assert Path("outputs/05-litert-export.png").exists()
    assert Path("previews/05-litert-export.png").exists()
