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

#parameters 
batch_size = 32
img_height = 200
img_width = 200

def create_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

# Code voor grafieken, info visualiseren.
def info(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def test(model, url, img_height, img_width, class_names):
  sunflower_url = url
  sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

  img = tf.keras.utils.load_img(
      sunflower_url, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )  

def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Location of where the model (weights) are saved 
checkpoint_path = "AI-Dev/checkpoints"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Make a dataset from dir 
data_dir = "AI-Dev/planten"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    shuffle=True,
    follow_links=False,
    crop_to_aspect_ratio=False
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    shuffle=True,
    follow_links=False,
    crop_to_aspect_ratio=False
)
class_names = train_ds.class_names

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
image_batch, labels_batch = next(iter(normalized_ds))

num_classes = len(class_names)

# Create model
model = create_model()

# Preparations for checkpoints (weights)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 #save_freq=5*batch_size
                                                 )

#Loading of weigths 
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model.load_weights(latest)

#training of model
epochs = 1

model.summary()
model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[cp_callback])

# Saving of weights 
# print("Saving checkpoint weights")
# model.save_weights(checkpoint_dir)

# Save model 
print("Saving model at AI-Dev/saved_model")
model.save('AI-Dev/saved_model')

#test accuracy 
loss, acc = model.evaluate(train_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#Testing of model
test(model, "AI-Dev/testFoto/Screenshot 2022-12-06 at 18.15.47.png", img_height, img_width, class_names)
