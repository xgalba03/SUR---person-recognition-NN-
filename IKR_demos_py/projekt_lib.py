from glob import glob

import cv2
import numpy as np
from scipy.io import wavfile
from ikrlib import mfcc
from imageio import imread


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


def png_load(dir_name):
    data = []
    file_names = []
    for f in glob(dir_name + '/*.png'):
        print('Processing file: ', f)
        file_names.append(f)
        # features.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float64))
        #img = cv2.imread(f, cv2.IMREAD_COLOR).astype(np.float64)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        img = img / 255
        data.append(img)

    return np.array(data), np.array(file_names)
