import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

print("=================================================")
print("    Face Ancestry - Fine Tuning (Retraining)     ")
print("=================================================")

# 1. Purana model load karein
model_path = 'models/face_ancestry_model.keras'
print("Purana AI Model load ho raha hai...")
model = tf.keras.models.load_model(model_path)

# 2. Naya data (Aapki tasweerein) kahan hai?
data_dir = 'data/corrections'

total_images = sum([len(files) for r, d, files in os.walk(data_dir)])
if total_images == 0:
    print("[Error]: Koi nayi tasweer nahi mili. Pehle live_predict.py chala kar tasweerein save karein.")
    exit()

print(f"[INFO]: Total {total_images} nayi tasweerein mili hain. Training shuru ho rahi hai...\n")

# 3. Data Augmentation (9 tasweeron se farzi variations banana)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Nayi classes ko pehchannay ke liye mapping wahi rakhni hai
classes = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=2, # Batch size chota hai kyunke data kam hai
    classes=classes, # Strict class mapping taake sequence kharab na ho
    class_mode='categorical',
    shuffle=True
)

# 4. Model ko fine-tune ke liye tayyar karein
# Learning rate (0.0001) bohot chota rakha hai taake purana knowledge delete na ho
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 5. Training Start!
print("\nAI aapke chehre se seekh raha hai...")
model.fit(
    train_data,
    epochs=10, # 10 rounds kafi honge in tasweeron ke liye
    verbose=1
)

# 6. Update shuda Model Save karein
model.save(model_path)
print(f"\n[SUCCESS]: Model successfully update ho gaya hai!")
print("Aapka AI ab pehle se zyada smart ho gaya hai.")