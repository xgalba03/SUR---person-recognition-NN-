import timeit
from os import path
from statistics import mean

import cv2
import numpy as np
import scipy
from keras.applications.densenet import layers
from keras.layers import Dropout
from keras.models import Sequential
from matplotlib import figure, pyplot as plt
from numpy.random import randint
from projekt_lib import png_load, eval_load
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import pandas as pd


def train_model():
    start = timeit.timeit()
    train_n = png_load('train_data/non_target_train', 50)
    train_t = png_load('train_data/target_train', 30)
    test_n = png_load('train_data/non_target_dev', 5)
    test_t = png_load('train_data/target_dev', 5)

    # label vector to determine class of img
    train_labels = np.r_[np.ones(len(train_t)), np.zeros(len(train_n))]
    test_labels = np.r_[np.ones(len(test_t)), np.zeros(len(test_n))]
    train_labels = tf.keras.utils.to_categorical(train_labels, 2)
    test_labels = tf.keras.utils.to_categorical(test_labels, 2)

    # merge classes
    train_data = np.r_[train_t, train_n]
    test_data = np.r_[test_t, test_n]
    """
    best yet
    model = Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(80, 80, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=40, activation='relu'))
    model.add(layers.Dense(units=2, activation='softmax'))
    """

    model = Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(80, 80, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=2, activation='softmax'))

    batch_size = 20  # 20
    steps = len(train_data) // batch_size
    epochs = 11  # 11

    model.summary()
    model.compile(
        optimizer=RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_data, train_labels,
        batch_size=batch_size,
        steps_per_epoch=steps,
        epochs=epochs,
        verbose=1,
        validation_data=(test_data, test_labels)
    )
    end = timeit.timeit()
    duration = end - start
    print("Training dataset size:", len(train_data))
    print("Model training time: ", duration)

    model.save('my_model')
    return model


if __name__ == "__main__":

    model = Sequential()
    if path.isdir('my_model'):
        model = tf.keras.models.load_model("my_model")

    else:
        model = train_model()

    eval_data, files = eval_load('eval')
    est = model.predict(eval_data)
    df = pd.DataFrame(est)
    df['filename'] = files
    print(est)
