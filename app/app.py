from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# -----------------------------
# Load model + class mapping
# -----------------------------
MODEL_PATH = "outputs/model.h5"
CLASS_MAP_PATH = "outputs/class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_MAP_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> class name
idx_to_class = {v: k for k, v in class_indices.items()}


# -----------------------------
# Preprocess function
# -----------------------------
def preprocess_image(img_path, img_size=224):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array


# -----------------------------
# Prediction route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file selected")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Preprocess
        img = preprocess_image(filepath)

        # Predict
        preds = model.predict(img)[0]
        class_id = np.argmax(preds)
        predicted_class = idx_to_class[class_id]
        confidence = float(np.max(preds) * 100)

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=f"{confidence:.2f}",
            image_path=filepath
        )

    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    app.run(debug=True)
