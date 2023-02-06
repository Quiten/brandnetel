from concurrent.futures import process
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import time 
import os 

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

saved_model_dir = "AI-Dev/saved_model"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)



