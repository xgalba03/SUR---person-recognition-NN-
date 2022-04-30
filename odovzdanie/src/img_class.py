from statistics import mean
import numpy as np
import scipy
from matplotlib import figure, pyplot as plt
from numpy.random import randint
from ikrlib import train_generative_linear_classifier
from projekt_lib import png2fea
from glob import glob

# Read images from target directories to 80x80 grayscale
train_n = png2fea('train_data/non_target_train').reshape(-1, 80 * 80)
train_t = png2fea('train_data/target_train').reshape(-1, 80 * 80)
test_n = png2fea('train_data/non_target_dev').reshape(-1, 80 * 80)
test_t = png2fea('train_data/target_dev').reshape(-1, 80 * 80)


# join classes to one matrix
x = np.r_[train_t, train_n]
# label vector to determine class of img
t = np.r_[np.ones(len(train_t)), np.zeros(len(train_n))]

mean_face = np.mean(x, axis=0)
x = x - mean_face
v, s, u = np.linalg.svd(x, full_matrices=False)
x = x.dot(u.T)

test_n_orig = test_n
test_t = (test_t - mean_face).dot(u.T)
test_n = (test_n - mean_face).dot(u.T)

D = 10
w, w0, _ = train_generative_linear_classifier(x[:, :D], t)
scores_t = test_t[:, :D].dot(w) + w0
scores_n = test_n[:, :D].dot(w) + w0




