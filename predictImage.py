import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model('saved_model/my_model.keras')

# Define a dictionary to map class indices to view names
class_to_view = {
    0: 'eo_front',
    1: 'eo_front_smile',
    2: 'eo_front_wide_smile',
    3: 'eo_left',
    4: 'eo_left_smile',
    5: 'eo_right',
    6: 'eo_right_smile',
    7: 'io_front_occlusion',
    8: 'io_front_occlusion_upwork',
    9: 'io_left_occlusion',
    10: 'io_lower_arch',
    11: 'io_right_occlusion',
    12: 'io_upper_arch',
    13: 'o_xray'
}

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to predict the view of the image
def predict_view(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_view = class_to_view[predicted_index]
    return predicted_view

# Example usage
for i in range(1, 10):
    print(i)
    image_path = f'.testImage/{i}.JPG'
    predicted_view = predict_view(image_path)
    print("Predicted view:", predicted_view)

