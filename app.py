"""Local educational demo for a legacy face-image dataset classifier.

The categories in this historical model are dataset labels, not reliable facts
about an individual's ancestry, ethnicity or identity. Do not deploy publicly.
Images are processed in memory and are never stored by this app.
"""
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "face_ancestry_model.keras"
DATASET_LABELS = [
    "Black", "East Asian", "Indian", "Latino_Hispanic",
    "Middle Eastern", "Southeast Asian", "White",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MiB upload limit


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("The research model file is missing.")
    # Delay loading TensorFlow until explicitly requested, so GET works without it.
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        upload = request.files.get("file")
        if not upload or not upload.filename:
            error = "Choose a JPG or PNG image."
        else:
            try:
                with Image.open(BytesIO(upload.read())) as image:
                    if image.format not in ("JPEG", "PNG"):
                        raise ValueError("Only JPG and PNG formats are supported.")
                    pixels = np.asarray(
                        image.convert("RGB").resize((224, 224)),
                        dtype=np.float32,
                    ) / 255.0
                scores = get_model().predict(pixels[np.newaxis, ...], verbose=0)[0]
                if len(scores) != len(DATASET_LABELS):
                    raise ValueError("The model's output shape is incompatible.")
                index_ = int(np.argmax(scores))
                result = (
                    f"Research dataset class: {DATASET_LABELS[index_]} "
                    f"(uncalibrated score: {float(scores[index_]):.2f}). "
                    "This is not a determination of ancestry or ethnicity."
                )
            except (UnidentifiedImageError, ValueError, OSError) as exc:
                error = f"Could not process image: {exc}"
            except Exception:
                app.logger.exception("Model inference failed")
                error = "The research model could not be loaded or run."
    return render_template("index.html", result=result, error=error)


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "index.html", result=None,
        error="Image is too large; upload a file smaller than 4 MB.",
    ), 413


if __name__ == "__main__":
    # Local machine only. No debug server or publicly accessible interface.
    app.run(host="127.0.0.1", port=5000, debug=False)
