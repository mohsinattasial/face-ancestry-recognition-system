import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Trained model load karein
model_path = 'models/face_ancestry_model.keras'
model = tf.keras.models.load_model(model_path)

# Apne dataset ke hisaab se classes list karein 
# (Agar CSV mein classes alag hain toh isko update kar lein)
classes = ['Black', 'East Asian', 'Indian/South Asian', 'Latino', 'Middle Eastern', 'Southeast Asian', 'White']

def predict_face(img_path):
    # Image preprocessing
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    # Prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    result = classes[predicted_class_index]
    print(f"\n[Mohsin's AI Model Result]")
    print(f"Prediction: Yeh face '{result}' ancestry se milta hai.")
    print(f"Confidence (Yaqeen): {confidence:.2f}%\n")

# Testing ke liye image ka path dein (yeh image aapke main folder mein honi chahiye)
predict_face('test_photo.jpg')