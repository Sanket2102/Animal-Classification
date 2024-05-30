import streamlit as st
import requests

from io import BytesIO


from keras._tf_keras.keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

st.write("# Animal Classification (Cats :cat: vs Dogs :dog:)")
url = "https://media.istockphoto.com/id/1501348253/photo/golden-retriever-and-british-shorthair-cat-lying-together-under-blanket.webp?b=1&s=170667a&w=0&k=20&c=xkZG8SuK0fhVQ-TZzwu9iR9Rh8OcpMBIqcThhSkv-ZA="

response = requests.get(url)
response.raise_for_status()  # Ensure we got a valid response
img = Image.open(BytesIO(response.content))

st.write(img)
st.write('''I developed a web application that utilizes a machine learning model to classify images of cats and dogs. This application allows users to upload images, which are then processed by a pre-trained model to determine whether the image is of a cat or a dog. The model was trained using Google Teachable Machine with a diverse set of labeled cat and dog images to ensure high accuracy. The user-friendly interface of the app makes it easy for users to interact with the model and receive instant predictions. This project showcases the practical application of machine learning in image classification, providing a seamless and efficient tool for users to differentiate between cat and dog images.''')

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




# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


with st.expander("To download test data for checking the model, click here"):
    # drive link for downloading test dataset
    st.write("Download here- https://drive.google.com/drive/u/2/folders/1w8xAM3aoF6ceUvbBtWShayvN9w0biFnp")

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
    st.write("Class:", class_name[2:])
    st.write("Confidence Score:", confidence_score)
