import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.preprocessing import image
from keras.applications.resnet import ResNet50, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser
from keras.utils import to_categorical
import csv

tf.keras.backend.clear_session()

# Configuration
epochs = 100
lr = 0.001

# Use GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU is available")
else:
    print("No GPU found, running on CPU")

# Set seed for reproducibility
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

# Function to load and preprocess the images
def load_train_images(train_path, size=(224, 224, 3)):
    images = []
    labels = []
    count = 0
    class_directories = sorted(os.listdir(train_path), key=lambda x: int(x))

    for folder in class_directories:
        sub_path = os.path.join(train_path, folder)
        print(sub_path)
        for img_file in os.listdir(sub_path):
            img_path = os.path.join(sub_path, img_file)

            img = image.load_img(img_path, target_size=size)
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)

            images.append(img_array)
            labels.append(count)

        count += 1

    return np.array(images), np.array(labels)

def load_test_images(test_path, size=(224, 224, 3)):
    images = []

    # Get a list of image files and sort them numerically
    image_files = sorted(os.listdir(test_path), key=lambda x: int(x.split('.')[0]))

    for img_file in image_files:
        img_path = os.path.join(test_path, img_file)

        img = image.load_img(img_path, target_size=size)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)

        images.append(img_array)

    return np.array(images)

# Load and preprocess the training data
train_path = "data/train"
X_train, y_train = load_train_images(train_path)

# Load and preprocess the test data
test_path = "data/test"
X_test = load_test_images(test_path)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.7, 1.3],
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    preprocessing_function=get_random_eraser(v_l=0, v_h=255),
)

datagen.fit(X_train)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_train_encoded = to_categorical(y_train, num_classes=len(label_encoder.classes_))

# Mixup generator
training_generator = MixupGenerator(X_train, y_train_encoded, batch_size=64, alpha=0.2, datagen=datagen)()

# Define the input shape for ResNet101
input_shape = (224, 224, 3)

# Model creation
def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-05))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(label_encoder.classes_), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

# Create and compile the model
model = create_model()

history = model.fit_generator(
    generator=training_generator,
    validation_data=(X_val, to_categorical(y_val, num_classes=len(label_encoder.classes_))),
    epochs=epochs,
    steps_per_epoch=len(X_train) // 64,
)

# Collect training history
training_history = history.history

# Create a CSV file to store training history
csv_columns = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
csv_file = 'training_history.csv'

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

    for epoch in range(epochs):
        row_data = {
            'epoch': epoch + 1,
            'loss': training_history['loss'][epoch],
            'accuracy': training_history['accuracy'][epoch],
            'val_loss': training_history['val_loss'][epoch],
            'val_accuracy': training_history['val_accuracy'][epoch]
        }
        writer.writerow(row_data)

# Evaluate the model on the validation data
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_val, y_pred_classes)
print(f"Validation Accuracy: {accuracy}")

model.save("model.h5")
