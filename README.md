# Intel Image Classification Web App

This repository contains a complete scaffold to train a convolutional neural network on the Intel Image Classification dataset and serve predictions through a Flask web interface.

What’s included
- data/ (instructions) — how to obtain and prepare the Intel dataset
- train.py — transfer-learning training script (Keras / TensorFlow)
- app/ — Flask app and model-serving utilities
  - app/__init__.py
  - app/routes.py
  - app/models.py
  - app/utils.py
  - templates/index.html
  - templates/result.html
  - static/style.css
- requirements.txt — Python dependencies
- Dockerfile — containerize the Flask app
- outputs/ — saved model path: `outputs/model.h5` (created when training)

Quick start (local)
1. Create virtual environment and install deps:
   python -m venv venv
   source venv/bin/activate or .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt

2. Prepare dataset
   
    Download the Intel Image Classification dataset (see data/download_instructions.md) and place extracted folders under `data/Intel/` so that subfolders `seg_train/seg_train/<class>/...` and `seg_test/seg_test/<class>/...` exist.
    # Dataset https://www.kaggle.com/datasets/puneet6060/intel-image-classification
    # Check training classes
       Get-ChildItem -Directory data\Intel\seg_train\seg_train
    # Check test classes
       Get-ChildItem -Directory data\Intel\seg_test\seg_test

    Run the preprocessing and optionally create train/val splits (the training script will use ImageDataGenerator and can point to `data/Intel/seg_train`).

4. Train model:
  python train.py --data-dir data/Intel/seg_train/seg_train --val-dir data/Intel/seg_test/seg_test --epochs 20 --batch-size 32 --out outputs

   The trained model will be saved at `outputs/model.h5`.

5. Run the Flask app:
   export FLASK_APP=app or $env:FLASK_APP='app'
   export FLASK_ENV=development or $env:FLASK_ENV='development'
   python -m app.routes

   Open http://127.0.0.1:5000/ and upload an image to classify.

Docker
- Build container:
  docker build -t intel-image-classifier .
- Run (model should be present in outputs/model.h5; mount outputs into the container):
  docker run -p 5000:5000 -v $(pwd)/outputs:/app/outputs intel-image-classifier

Notes & next steps
- Add Grad-CAM explainability to show salient regions.
- Replace ImageDataGenerator with tf.data pipeline for performance.
- Add model-versioning and a small REST endpoint to return top-k predictions in JSON.
