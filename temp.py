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
print(f'TensorFlow version: {tf.__version__}')

# Print model summary to ensure it's loaded correctly
model.summary()

model.save('new_model.h5')
