import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
import pandas as pd

# Check and set GPU memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU is available")
else:
    print("No GPU found, running on CPU")

# Load the saved model
model = load_model("model.h5")

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

# Load and preprocess the test data
test_path = "data/test"
X_test = load_test_images(test_path)

# Generate predictions for the test data using the loaded model
test_predictions = model.predict(X_test)
test_predictions_classes = np.argmax(test_predictions, axis=1)

# Create a submission.csv file
submission_df = pd.DataFrame({
    'ID': [f"{i}.jpg" for i in range(len(test_predictions_classes))],
    'Label': test_predictions_classes
})

submission_df.to_csv("submission.csv", index=False)
