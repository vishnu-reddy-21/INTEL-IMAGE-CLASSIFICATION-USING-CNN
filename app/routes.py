#!/usr/bin/env python3
"""
Flask app routes: upload UI and prediction API.
Run with: python -m app.routes
"""
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from pathlib import Path
from werkzeug.utils import secure_filename
import os
from .models import load_model, predict
from .utils import read_image_from_bytes

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED = {"png","jpg","jpeg"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.secret_key = "dev-key"

# load model at startup if available (no-op if not trained yet)
try:
    load_model()
except Exception:
    pass

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload_predict():
    if 'file' not in request.files:
        flash("No file part")
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = UPLOAD_FOLDER / filename
        file.save(save_path)
        try:
            pil = read_image_from_bytes(save_path.read_bytes())
            preds = predict(pil, top_k=1)
            top_class, top_prob = preds[0]
        except Exception as e:
            flash(f"Prediction failed: {e}")
            return redirect(url_for('index'))
        return render_template("result.html", filename=filename, predicted_class=top_class, accuracy=top_prob)
    else:
        flash("Unsupported file type")
        return redirect(url_for('index'))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)