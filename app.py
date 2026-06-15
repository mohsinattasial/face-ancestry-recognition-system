from flask import Flask, request, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Tasweerein save karne ki jagah (Folder path)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# 1. AI Model Load karein
print("Web Server AI Model load kar raha hai...")
model = tf.keras.models.load_model('models/face_ancestry_model.keras')
classes = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
print("Model Load ho gaya!")

# 2. Website ka Main Page
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None
    
    if request.method == 'POST':
        # Agar user ne koi file upload ki hai
        file = request.files['file']
        if file and file.filename != '':
            # Photo ko static/uploads mein save karein
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # HTML mein tasweer dikhane ke liye path
            image_path = f"/{filepath}"

            # AI Prediction
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = model.predict(img_array)[0]
            pred_idx = np.argmax(predictions)
            conf = np.max(predictions) * 100

            # Result format
            result = f"{classes[pred_idx]} ({conf:.1f}%)"

    return render_template('index.html', result=result, image_path=image_path)

if __name__ == '__main__':
    # Localhost par server on karein
    app.run(debug=True, port=5000)