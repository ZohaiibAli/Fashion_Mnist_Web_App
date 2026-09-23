"""Backend logic: model loading, data loading, preprocessing, and prediction."""

from pathlib import Path

import numpy as np
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "fashion_mnist.keras"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from tensorflow import keras

    return keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner="Loading Fashion MNIST test set...")
def load_test_data():
    from tensorflow.keras.datasets import fashion_mnist

    (_, _), (test_images, test_labels) = fashion_mnist.load_data()
    return test_images, test_labels


@st.cache_data(show_spinner="Evaluating model on the test set...")
def compute_test_metrics():
    """Runs the cached model against the full test set and returns accuracy
    plus a per-class precision/recall/f1 report."""
    from sklearn.metrics import classification_report

    model = load_model()
    test_images, test_labels = load_test_data()
    flattened = test_images.reshape((-1, 784)).astype("float32") / 255.0
    probs = model.predict(flattened, verbose=0)
    predicted = np.argmax(probs, axis=1)
    report = classification_report(
        test_labels, predicted, target_names=CLASS_NAMES, output_dict=True
    )
    accuracy = report["accuracy"]
    return accuracy, report


def preprocess_test_image(image_28x28: np.ndarray) -> np.ndarray:
    """Flattens and normalizes a raw 28x28 uint8 array from the dataset."""
    return image_28x28.astype("float32").reshape(1, 784) / 255.0


def preprocess_pil_image(pil_image, invert: bool = True):
    """Converts an arbitrary uploaded image into the 28x28 grayscale, flattened,
    normalized form the model expects. Returns (model_input, display_28x28)."""
    gray = pil_image.convert("L").resize((28, 28))
    arr = np.array(gray).astype("float32")
    if invert:
        arr = 255.0 - arr
    display = arr.astype("uint8")
    model_input = (arr / 255.0).reshape(1, 784)
    return model_input, display


def predict(model, model_input: np.ndarray) -> np.ndarray:
    """Returns the class probability vector for a single (1, 784) input."""
    return model.predict(model_input, verbose=0)[0]
