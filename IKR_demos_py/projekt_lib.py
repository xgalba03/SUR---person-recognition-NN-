from glob import glob
from random import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from ikrlib import mfcc
from imageio import imread
import albumentations as A


def test():
    print("hey lib")


def wav16khz2mfcc(dir_name):
    """
    Loads all *.wav files from directory dir_name (must be 16kHz), converts them into MFCC
    features (13 coefficients) and stores them into a dictionary. Keys are the file names
    and values and 2D numpy arrays of MFCC features.
    """
    features = {}
    for f in glob(dir_name + '/*.wav'):
        print('Processing file: ', f)
        rate, s = wavfile.read(f)
        assert (rate == 16000)
        features[f] = mfcc(s, 400, 240, 512, 16000, 23, 13)
    return features


def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    data = []
    for f in glob(dir_name + '/*.png'):
        print('Processing file: ', f)
        data.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float64))
    return np.array(data)


def display_image(im_data):
    dpi = 80
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()


def augment_img(image, major=False):
    if major:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=(0, 0.1), rotate_limit=45, p=0.50),
            # A.Blur(blur_limit=3),
            A.OneOf([
                A.GaussNoise(p=0.3, mean=50, var_limit=(10, 100)),
                A.MultiplicativeNoise(p=0.5, multiplier=(0.7, 1.5), per_channel=True, elementwise=True)
            ])
        ])
    else:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=(0, 0.1), rotate_limit=45, p=0.50),
        ])
    augmented_image = transform(image=image)['image']
    return augmented_image


def png_load(dir_name, augment_count=0):
    data = []
    for f in glob(dir_name + '/*.png'):
        # print('Processing file: ', f)
        # features.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float64))
        # img = cv2.imread(f, cv2.IMREAD_COLOR).astype(np.float64)
        # img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        image = cv2.imread(f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = image / 255
        data.append(img)

        # display_image(image)

        for x in range(augment_count):
            if x > (augment_count // 2):
                augmented_image = augment_img(image, major=True)

            else:
                augmented_image = augment_img(image)
            img_a = augmented_image / 255
            data.append(img_a)
            # display_image(augmented_image)

    return np.array(data)


def eval_load(dir_name):
    data = []
    file_names = []
    for f in glob(dir_name + '/*.png'):
        # print('Processing file: ', f)
        file_names.append(f)
        # features.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float64))
        # img = cv2.imread(f, cv2.IMREAD_COLOR).astype(np.float64)
        # img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        image = cv2.imread(f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = image / 255
        data.append(img)

    return np.array(data), np.array(file_names)
