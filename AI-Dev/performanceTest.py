import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import time 
import os 
from test2 import test, training, info

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# Location of where the model (weights) are saved 
checkpoint_path = "/Users/mr.q/Desktop/School/PWS/coderclass/AI-Dev/checkpointsPlant/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# dataset_url = "/Users/mr.q/Desktop/School/PWS/coderclass/AI-Dev/"
# data_dir = tf.keras.utils.get_file('planten', origin=dataset_url, untar=True)
# data_dir = pathlib.Path(data_dir)

# Make a dataset from dir 
data_dir = "/Users/mr.q/Desktop/School/PWS/coderclass/AI-Dev/planten"

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    follow_links=False,
    crop_to_aspect_ratio=False
)

#parameters 
batch_size = 32
img_height = 180
img_width = 180

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

# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)
# print(class_names)


# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

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

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#   for i in range(9):
#     augmented_images = data_augmentation(images)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0].numpy().astype("uint8"))
#     plt.axis("off")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

# Create model
def create_model():
  model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = create_model()

# Preparations for checkpoints (weights)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 #save_freq=5*batch_size
                                                 )

#training of model
history = training(model, 5)
info(history, 5)

# model.fit(
#     train_ds,
#     validation_data=(val_ds),
#     epochs=5,
#     callbacks=[cp_callback])

#Saving of weights 
# history.save(checkpoint_dir)

#Loading of weigths 
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

#test accuracy 
# loss, acc = model.evaluate(train_ds, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#Testing of model
# test(model, "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg")

