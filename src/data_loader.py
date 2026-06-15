import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Sahi path (Sirf 'data/' aayega taake error na aaye)
train_csv_path = 'data/train_labels.csv'
train_img_dir = 'data/' 

# CSV read karein
train_df = pd.read_csv(train_csv_path)

# Data Augmentation & Scaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)

print("Training Data Load ho raha hai...")

# Generator setup
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_img_dir,
    x_col='file',        
    y_col='race',        
    target_size=(224, 224), 
    batch_size=32,
    class_mode='categorical'
)