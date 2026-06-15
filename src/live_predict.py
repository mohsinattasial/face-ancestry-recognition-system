import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time

print("=================================================")
print(" Live Face Ancestry & Feedback System - Started  ")
print("=================================================")

# 1. AI Model aur Classes
model_path = 'models/face_ancestry_model.keras'
print("AI Model load ho raha hai...")
model = tf.keras.models.load_model(model_path)
classes = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']

# 2. Feedback (Corrections) Folders Setup
# Agar folders nahi hain, toh yeh code khud bana dega
base_dir = "data/corrections"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for class_name in classes:
    folder_path = os.path.join(base_dir, class_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# 3. User Instructions
print("\n[INSTRUCTIONS FOR CONTINUOUS LEARNING]")
print("Agar AI ghalat bataye, toh apne keyboard se yeh number dabayein taake tasweer us label mein save ho jaye:")
for i, c in enumerate(classes):
    print(f" Press '{i}' for {c}")
print("Band karne ke liye 'q' dabayein.\n")

# 4. OpenCV Camera Setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Keyboard ka button check karein (1 millisecond delay ke sath)
    key = cv2.waitKey(1) & 0xFF

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            # AI ke liye image pre-process karein
            face_roi_resized = cv2.resize(face_roi, (224, 224))
            img_array = img_to_array(face_roi_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  
            
            # AI Prediction
            predictions = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(predictions[0])
            conf = np.max(predictions[0]) * 100
            
            # Result screen par likhein
            result_text = f"{classes[pred_idx]} ({conf:.1f}%)"
            cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- DATA COLLECTION LOGIC ---
            # Agar user ne 0 se 6 ke darmiyan koi button dabaya hai
            if ord('0') <= key <= ord('6'):
                correct_class_idx = int(chr(key))
                correct_class_name = classes[correct_class_idx]
                
                # Nayi tasweer ka naam timestamp ke sath banayein taake overwrite na ho
                filename = f"{correct_class_name}_{int(time.time())}.jpg"
                filepath = os.path.join(base_dir, correct_class_name, filename)
                
                # Tasweer save karein
                cv2.imwrite(filepath, face_roi)
                print(f"[SUCCESS] Tasweer Save Ho Gayi: {filepath}")
                
                # Screen par 'SAVED!' likha show karein (Red color mein)
                cv2.putText(frame, "SAVED!", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        except Exception as e:
            pass

    cv2.imshow('Live Face Ancestry AI', frame)

    # 'q' dabane par camera band ho jayega
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nLive Camera band kar diya gaya hai.")