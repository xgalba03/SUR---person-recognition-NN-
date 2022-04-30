from glob import glob
from random import random

import cv2
import numpy as np
import librosa
from matplotlib import pyplot as plt
from scipy.io import wavfile
from ikrlib import mfcc
from imageio import imread
import albumentations as A


def process_audio(dir_name):
    file_names = []
    features = {}
    for f in glob(dir_name + '/*.wav'):
        print('Processing file: ', f)
        file_names.append(f)
        y, sr = librosa.load(f)
        y_trim = librosa.effects.remix(y, intervals=librosa.effects.split(y))
        features[f] = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=13).T
    return features, np.array(file_names)

def train_gmm(x, ws, mus, covs):
    gamma = np.vstack([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)])
    logevidence = logsumexp(gamma, axis=0)
    gamma = np.exp(gamma - logevidence)
    gammasum = gamma.sum(axis=1)
    ws = gammasum / len(x)
    mus = gamma.dot(x) / gammasum[:, np.newaxis]

    if covs[0].ndim == 1:
        covs = gamma.dot(x ** 2) / gammasum[:, np.newaxis] - mus ** 2
    else:
        covs = np.array(
            [(gamma[i] * x.T).dot(x) / gammasum[i] - mus[i][:, newaxis].dot(mus[[i]]) for i in range(len(ws))])
    return ws, mus, covs


def logpdf_gmm(x, ws, mus, covs):
    return logsumexp([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)], axis=0)

def logpdf_gauss(x, mu, cov):
    assert (mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1]))
    x = np.atleast_2d(x) - mu
    if cov.ndim == 1:
        return -0.5 * (len(mu) * np.log(2 * pi) + np.sum(np.log(cov)) + np.sum((x ** 2) / cov, axis=1))
    else:
        return -0.5 * (len(mu) * np.log(2 * pi) + np.linalg.slogdet(cov)[1] + np.sum(x.dot(inv(cov)) * x, axis=1))


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
