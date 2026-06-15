import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Main Tkinter window ko background mein hide karne ke liye
root = tk.Tk()
root.withdraw()

print("=================================================")
print("  Face Ancestry Recognition - Interactive Demo  ")
print("  Developed by: Mohsin Atta (Data Science, IUB)  ")
print("=================================================")

# 1. Trained model load karne ka path
model_path = 'models/face_ancestry_model.keras'

if not os.path.exists(model_path):
    print(f"\n[Error]: Model file '{model_path}' nahi mili!")
    print("Kripya check karein ke 'models' folder mein 'face_ancestry_model.keras' mojood hai.")
    input("\nExit karne ke liye Enter dabayein...")
    exit()

print("\nAI Model Load ho raha hai... Kripya thora intezar karein...")
model = tf.keras.models.load_model(model_path)
print("Model successfully load ho gaya!\n")

# 2. Dataset classes mapping
classes = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']

# 3. Interactive File Selection Dialog (Windows Popup)
print("[PROMPT]: Ek window khuli hai, wahan se apni pasand ki koi bhi photo select karein...")
file_path = filedialog.askopenfilename(
    title="Testing Ke Liye Photo Select Karein",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
)

# 4. Prediction Logic
if not file_path:
    print("\n[Warning]: Aapne koi photo select nahi ki. Script band ho rahi hai.")
else:
    print(f"\nSelected Photo: {file_path}")
    print("AI Model photo ko scan aur analyze kar raha hai...")
    
    try:
        # Image preprocessing (Resizing to 224x224 and normalization)
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        # Model Prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        result = classes[predicted_class_index]
        
        # Beautiful results representation
        print("\n=====================================")
        print("      [Mohsin's AI Model Result]      ")
        print("=====================================")
        print(f"Prediction: Yeh face '{result}' ancestry se match karta hai.")
        print(f"Confidence (Yaqeen): {confidence:.2f}%")
        print("=====================================\n")
        
    except Exception as e:
        print(f"\n[Error]: Photo process karne mein koi masla aaya hai: {e}")

# Script terminal ko open rakhne ke liye
input("Demo khatam! Window band karne ke liye Enter dabayein...")