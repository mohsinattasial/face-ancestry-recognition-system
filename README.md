# Experimental Face-Image Dataset Classifier

This repository is an educational project using a previously trained image classifier.
**Ancestry, race and ethnicity cannot be reliably determined from facial appearance.**
Its dataset-class outputs are neither identity verification nor evidence about an individual.

## Local demo
Use a Python environment compatible with the checked-in TensorFlow model and install
the dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python app.py
```

Visit `http://127.0.0.1:5000`. The Flask app uses the new
`templates/index.html`, processes consenting JPG/PNG uploads in memory,
limits upload size and does **not** enable Flask debug mode or public network
binding. No uploaded photos are stored by the app.

The model is only loaded when an image is submitted. If TensorFlow, the saved
model, or required packages are incompatible, the page reports an error;
that does not mean the model has been scientifically validated.

## Code
- `app.py`: local demonstration.
- `templates/index.html`: user interface.
- `src/`: original training and local experimental scripts.
- `models/face_ancestry_model.keras`: legacy model artifact.
- `tests/`: basic app route tests.
- `read me.md`: older project notes (not independent evidence of accuracy).

## Limitations
No representative independent validation or bias audit is provided. Never use
this project for identification, eligibility, employment, diagnosis or other
consequential decisions. Use only consented images. Original training scripts
and model assets remain research artifacts.

Copyright © 2026 Mohsin Atta and contributors. Code: MIT license;
third-party data and model components may have different terms.
