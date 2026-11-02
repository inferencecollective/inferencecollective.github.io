"""Alias for Lab 03 Keras CNN Image Classifier."""
from labs.lab03_keras_cnn import run

if __name__ == "__main__":
    res = run()
    print(f"Lab 03 completed: test_acc={res['test_accuracy']}, backend={res['backend']}")
