from glob import glob

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
    features = {}
    for f in glob(dir_name + '/*.png'):
        print('Processing file: ', f)
        features[f] = imread(f, True).astype(np.float64)
    return features