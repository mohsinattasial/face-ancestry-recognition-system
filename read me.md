# Face Ancestry Recognition System — Experimental Coursework

An educational TensorFlow/Keras computer-vision repository with training and prediction scripts. **Ancestry, ethnicity and race cannot be reliably determined from facial appearance.** Any labels produced by this model are not facts about a person and should not be used for identity, eligibility, hiring, medical or other consequential decisions.

## Repository structure
- `src/data_loader.py`, `src/model.py`, `src/train.py`, `src/fine_tune.py`: experimentation/training.
- `src/predict.py`, `src/predict_select.py`, `src/live_predict.py`: experimental inference utilities.
- `models/face_ancestry_model.keras`: serialized model asset.
- `app.py`: incomplete Flask proof-of-concept; its referenced `templates/index.html` is not present in this repository.
- `requirements.txt`: development dependencies.

## Important issues and limitations
- The former README's approximately 40.5% **training** accuracy does not measure generalization; a reproducible held-out evaluation is missing.
- Images and labels can reflect biased collection choices and are not valid proxies for ancestry.
- Existing upload handling needs file validation, random filenames and deletion policies; **do not deploy `app.py` publicly** in its current form.
- Use only consented example images and remove personal photos from repositories.
- Generated `__pycache__` and user-uploaded images should not be committed.

This is a learning artifact, not a validated ancestry inference service.

Author: Mohsin Atta.
