import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# need to install CUDA for the prog to run faster

# Define paths to your dataset
train_dir = 'data/train'
val_dir = 'data/validation'

# Load and preprocess data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # or 'binary' if 2 classes
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # or 'binary' if 2 classes
    subset='validation'
)

# Load pre-trained VGG16 model and fine-tune
base_model = VGG16(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Add new top layers for dental image classification
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Use 'sigmoid' if binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Create the directory if it doesn't exist
save_dir = 'saved_model'
os.makedirs(save_dir, exist_ok=True)

# Save the model in SavedModel format
# model.save('saved_model/', save_format='tf')
model.save('saved_model/my_model.keras')
