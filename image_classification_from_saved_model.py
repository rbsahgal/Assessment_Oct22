import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

# Dataset details 
image_height = 150
image_width = 150
number_of_channels = 3
number_of_classes = 6

#Hyper-Parameters
batch_size = 128
NUMBER_OF_EPOCHS = 1
LEARNING_RATE = 0.001

print("############-------- START --------############")

print("This code run on on Python version : ", sys.version)
# Print statement check tensorflow is running
print("The Tensorflow version used is : " + tf. __version__)

# Load Image Data from local computer
train_set = tf.keras.preprocessing.image_dataset_from_directory(r"E:\Work\vs_code\Assessment_Oct22\SceneryDataset\seg_train",
color_mode= "rgb",
batch_size=batch_size,
image_size=(image_height, image_width),
shuffle=True,
seed=123)


validation_set = tf.keras.preprocessing.image_dataset_from_directory(r"E:\Work\vs_code\Assessment_Oct22\SceneryDataset\seg_test",
color_mode= "rgb",
batch_size=batch_size,
image_size=(image_height, image_width),
shuffle=True,
seed=123)

print(train_set.class_names)
print(validation_set.class_names)
def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


class_names = train_set.class_names
for image_batch, labels_batch in train_set:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


AUTOTUNE = tf.data.AUTOTUNE
#BATCH_SIZE = 32

for image_batch, labels_batch in train_set:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
validation_set = validation_set.cache().prefetch(buffer_size=AUTOTUNE)


model = keras.models.load_model('saved_model/')


history = model.fit(
  train_set,
  validation_data=validation_set,
  epochs=NUMBER_OF_EPOCHS
)

model.evaluate(validation_set, verbose=2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(NUMBER_OF_EPOCHS)

plt.figure(1,figsize=(8, 8))
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


# Predict on new images
pred_img = tf.keras.utils.load_img(
    r"E:\Work\vs_code\Assessment_Oct22\SceneryDataset\seg_pred\9992.jpg", target_size=(image_height, image_width), color_mode="rgb"
)
plt.figure(2,figsize=(5, 5))
plt.imshow(pred_img)

img_array = tf.keras.utils.img_to_array(pred_img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
prediction_list = model.predict(img_array) 
prediction = prediction_list[0]
predictionIndex = np.argmax(prediction)
predictedClass = class_names[predictionIndex]
prediction_confidence = round (100 * np.max(prediction), 2)

for prediction_vals in prediction_list:
    print(prediction_vals)
    print("/n")

title_string  = "This image is {} with a {:.2f} % confidence". format(predictedClass, prediction_confidence)
plt.title(title_string)
plt.axis('off')
plt.show()

model.save('saved_model/')

