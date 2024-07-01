import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import os

# Define constants
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7
DATASET_PATH = '/home/ubuntu/human-emotions-capturing/dataset'  # Update this path to the location of the dataset

# Function to load and preprocess the dataset
def load_data(dataset_path):
    data = []
    labels = []
    for emotion in os.listdir(dataset_path):
        emotion_path = os.path.join(dataset_path, emotion)
        if os.path.isdir(emotion_path):
            for img_file in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_file)
                img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = img / 255.0  # Normalize the image
                data.append(img)
                labels.append(emotion)
    data = np.array(data)
    labels = pd.get_dummies(labels).values
    return data, labels

# Load and preprocess the dataset
data, labels = load_data(DATASET_PATH)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr])

# Save the model
model.save('emotion_recognition_model.h5')

# Save the training history
np.save('/home/ubuntu/human-emotions-capturing/training_history.npy', history.history)
