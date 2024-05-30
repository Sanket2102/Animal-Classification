import streamlit as st

st.write("Hello")

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D class to handle the unrecognized parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

# Register the custom object when loading the model
custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}

# Load the model
model = load_model('keras_model.h5', custom_objects=custom_objects)

# Print TensorFlow version
st.write(f'TensorFlow version: {tf.__version__}')

from keras._tf_keras.keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

uploaded_files = st.file_uploader("Choose an image...", accept_multiple_files=False, type=["jpg", "jpeg", "png"])

if uploaded_files:
    image = Image.open(uploaded_files).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    st.write(image)
    # turn the image into a numpy array
    image_array = np.asarray(image)

    # # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    st.write("Class:", class_name[2:], end="")
    st.write("Confidence Score:", confidence_score)
