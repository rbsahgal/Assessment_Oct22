import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

# Dataset details 
image_height = 150
image_width = 150
number_of_channels = 3
number_of_classes = 6

#Hyper-Parameters
batch_size = 256
NUMBER_OF_EPOCHS = 15
LEARNING_RATE = 0.002

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

#labels='inferred', 
#label_mode="categorical",
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


AUTOTUNE = tf.data.AUTOTUNE

image_augmentation = keras.Sequential(
    [       
        layers.RandomFlip(mode="horizontal",
                        input_shape=(image_height,
                                    image_width,
                                    3)),
        layers.RandomContrast(factor=0.1,),
        layers.RandomBrightness(factor=0.15),
        layers.RandomRotation(0.1)
    ]
)

class_names = train_set.class_names
for image_batch, labels_batch in train_set:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
validation_set = validation_set.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential([
  image_augmentation,
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3,padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3,padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3,padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3,padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  layers.Dropout(0.15),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(number_of_classes)
])



model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

def lr_scheduler(epoch, lr):
  if epoch < 3:
    return lr
  else:
    return lr * 0.999
    


lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
#tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_req=1)

history = model.fit(
  train_set,
  validation_data=validation_set,
  epochs=NUMBER_OF_EPOCHS,
  callbacks=[lr_scheduler_callback]
)

model.save(r"E:\Work\vs_code\Assessment_Oct22\saved_model")
model.evaluate(validation_set, verbose=2)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(NUMBER_OF_EPOCHS)



# Plot training and validation graphs
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
score = tf.nn.softmax(prediction_list[0])
predictionIndex = np.argmax(prediction)
predictedClass = class_names[predictionIndex]
prediction_confidence = round (100 * np.max(prediction), 2)

for prediction_vals in prediction_list:
    print(prediction_vals)
    print("/n")

title_string  = "This image is {} with a {:.2f} % confidence". format(class_names[np.argmax(score)], 100 * np.max(score))
plt.title(title_string)
plt.axis('off')
plt.show()






