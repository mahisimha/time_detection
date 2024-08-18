import os
import csv
import random
import math
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Enable mixed precision
try:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
except ImportError:
    # If you are using TensorFlow 2.4.0 and above, use this
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')


labels_df = pd.read_csv('labels.csv')
labels_df['path'] = labels_df['Image Filename'].apply(lambda x: os.path.join('images_tf', x))

# Parameters
IMG_SIZE = (256,256)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2

# ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1/255.,
    validation_split=VALIDATION_SPLIT
)

# Creating training and validation generators
train_generator = datagen.flow_from_dataframe(
    labels_df,
    x_col='path',
    y_col=['hour', 'minute'],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='multi_output',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    labels_df,
    x_col='path',
    y_col=['hour', 'minute'],
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='multi_output',
    subset='validation'
)

def build_custom_cnn(input_shape):
    inputs = Input(shape=input_shape)
   
    x = Conv2D(16, (3, 3), activation='leaky_relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='leaky_relu', padding='same')(x)
    x = BatchNormalization()(x)
    # Convolutional block 3
    x = Conv2D(16, (3, 3), activation='leaky_relu', padding='same')(x)
    x = BatchNormalization()(x) 
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='leaky_relu')(x)
    x = Dropout(0.5)(x)
    # Output layers for hour and minute
    hour_output = Dense(12, activation='softmax', name='hour_output')(x)
    minute_output = Dense(60, activation='softmax', name='minute_output')(x)

    return Model(inputs, [hour_output, minute_output])

# Build the model
model = build_custom_cnn(IMG_SIZE + (3,))
model.compile(
    optimizer='SGD',
    loss={'hour_output': 'sparse_categorical_crossentropy', 'minute_output': 'sparse_categorical_crossentropy'},
    metrics={'hour_output': 'accuracy', 'minute_output': 'accuracy'}
)

model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)
model.evaluate(validation_generator, batch_size=16)



# Save the model
model.save('clock_time_predictor.h5')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['hour_output_accuracy'], label='Hour Accuracy')
plt.plot(history.history['minute_output_accuracy'], label='Minute Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['val_hour_output_accuracy'], label='Val Hour Accuracy')
plt.plot(history.history['val_minute_output_accuracy'], label='Val Minute Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()