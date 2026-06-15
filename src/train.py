import os
from tensorflow.keras.optimizers import Adam
from data_loader import train_generator 
from model import build_model

print("-------------------------------------------------")
print("Face Ancestry Training Initialized - Mohsin Atta")
print("-------------------------------------------------")

# Total classes detect karein
num_classes = len(train_generator.class_indices)
print(f"Total Categories Found: {num_classes}")
print("Classes mapping:", train_generator.class_indices)

# Model setup
model = build_model(num_classes)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training start
print("Deep Learning training start ho rahi hai...")
epochs = 10 

history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.samples // train_generator.batch_size
)

# Model Save path
save_path = 'models/face_ancestry_model.keras'
os.makedirs('models', exist_ok=True)
model.save(save_path)

print(f"Model successfully saved at: {save_path}")