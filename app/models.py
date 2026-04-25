"""
Model loader and prediction helper.
Loads Keras model at startup and exposes predict(image_pil) -> list of (class, prob).
"""
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import json

_MODEL = None
_CLASS_IDX = None
_MODEL_PATH = Path("outputs/model.h5")
_CLASS_IDX_PATH = Path("outputs/class_indices.json")
_INPUT_SIZE = (224,224)

def load_model(model_path=None, class_idx_path=None):
    global _MODEL, _CLASS_IDX
    model_path = Path(model_path or _MODEL_PATH)
    class_idx_path = Path(class_idx_path or _CLASS_IDX_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    _MODEL = tf.keras.models.load_model(str(model_path))
    if class_idx_path.exists():
        with open(class_idx_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        # invert mapping: ensure keys are ints so lookups by index succeed
        _CLASS_IDX = {int(v): k for k, v in idx.items()}
    else:
        _CLASS_IDX = None
    return _MODEL

def preprocess_pil(image_pil, target_size=None):
    target_size = target_size or _INPUT_SIZE
    img = image_pil.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict(image_pil, top_k=3):
    if _MODEL is None:
        load_model()
    x = preprocess_pil(image_pil)
    probs = _MODEL.predict(x)[0]
    idx_sorted = np.argsort(probs)[::-1][:top_k]
    results = []
    for i in idx_sorted:
        key = int(i)
        label = _CLASS_IDX.get(key, str(key)) if _CLASS_IDX else str(key)
        results.append((label, float(probs[int(i)])))
    return results