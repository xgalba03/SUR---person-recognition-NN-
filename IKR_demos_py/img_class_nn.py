from statistics import mean

import cv2
import numpy as np
import scipy
from keras.applications.densenet import layers
from keras.models import Sequential
from matplotlib import figure, pyplot as plt
from numpy.random import randint
from projekt_lib import png_load
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import pandas as pd

# Read images from target directories to 80x80 grayscale
train_n,f = png_load('train_data/non_target_train')
train_t,f = png_load('train_data/target_train')
test_n,f = png_load('train_data/non_target_dev')
test_t,f = png_load('train_data/target_dev')

# label vector to determine class of img
train_labels= np.r_[np.ones(len(train_t)), np.zeros(len(train_n))]
test_labels= np.r_[np.ones(len(test_t)), np.zeros(len(test_n))]
train_labels= tf.keras.utils.to_categorical(train_labels, 2)
test_labels= tf.keras.utils.to_categorical(test_labels, 2)


train_data = np.r_[train_t, train_n]
test_data = np.r_[test_t, test_n]

model = Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(80, 80, 1)))
model.add(layers.AveragePooling2D())
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())
#model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
#model.add(layers.AveragePooling2D())
model.add(layers.Flatten())
#model.add(layers.Dense(units=300, activation='relu'))
model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dense(units=40, activation='relu'))
model.add(layers.Dense(units=2, activation='softmax'))

batch_size = 30  # 30
epochs = 10  # 15

model.summary()
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data, train_labels,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(test_data, test_labels)
)

eval_data, files = png_load('eval')
est = model.predict(eval_data)
df = pd.DataFrame(est)
df['filename'] = files
print(est)
