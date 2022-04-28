import numpy as np
import scipy
from matplotlib import pyplot as plt

from projekt_lib import wav16khz2mfcc,png2fea


non_target_wav_train = wav16khz2mfcc('train_data/non_target_train').values()
target_wav_train = wav16khz2mfcc('train_data/target_train').values()
non_target_wav_dev = wav16khz2mfcc('train_data/non_target_dev').values()
target_wav_dev = wav16khz2mfcc('train_data/target_dev').values()

# non_target_png_train = png2fea('train_data/non_target_train').values()
# target_png_train = png2fea('train_data/target_train').values()
# non_target_png_dev = png2fea('train_data/non_target_dev').values()
# target_png_dev = png2fea('train_data/target_dev').values()

train_target = np.vstack(target_wav_train)
train_non = np.vstack(non_target_wav_train)
dim = train_target.shape[1]

# PCA reduction to 2 dimensions

cov_tot = np.cov(np.vstack([train_non, train_target]).T, bias=True)
# take just 2 largest eigenvalues and corresponding eigenvectors
d, e = scipy.linalg.eigh(cov_tot, eigvals=(dim - 2, dim - 1))

train_n_pca = train_non.dot(e)
train_t_pca = train_target.dot(e)
plt.plot(train_n_pca[:, 1], train_n_pca[:, 0], 'b.', ms=1)
plt.plot(train_t_pca[:, 1], train_t_pca[:, 0], 'r.', ms=1)
plt.show()

#print(non_target_wav_train)