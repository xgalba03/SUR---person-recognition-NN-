import numpy as np

import scipy
from matplotlib import pyplot as plt

from numpy.random import randint
from ikrlib import train_gmm, logpdf_gmm
from projekt_lib import wav16khz2mfcc,png2fea


train_n = wav16khz2mfcc('train_data/non_target_train').values()
train_t = wav16khz2mfcc('train_data/target_train').values()
test_n = wav16khz2mfcc('train_data/non_target_dev').values()
test_t = wav16khz2mfcc('train_data/target_dev').values()

# non_target_png_train = png2fea('train_data/non_target_train').values()
# target_png_train = png2fea('train_data/target_train').values()
# non_target_png_dev = png2fea('train_data/non_target_dev').values()
# target_png_dev = png2fea('train_data/target_dev').values()

train_t = np.vstack(train_t)
train_n = np.vstack(train_n)

#two gmm models to train and test
M_t = 3
MUs_t = train_t[randint(1, len(train_t), M_t)]
#COVs_t = [np.var(train_t, axis=0)] * M_t
COVs_t = [np.cov(train_t.T)] * M_t
Ws_t = np.ones(M_t) / M_t

M_n = 20
MUs_n = train_n[randint(1, len(train_n), M_n)]
#COVs_t = [np.var(train_n, axis=0)] * M_n
COVs_n = [np.cov(train_n.T)] * M_n
Ws_n = np.ones(M_n) / M_n

# Run 30 iterations of EM algorithm to train the two GMMs from males and females
for jj in range(30):
    [Ws_t, MUs_t, COVs_t, TTL_t] = train_gmm(train_t, Ws_t, MUs_t, COVs_t);
    [Ws_n, MUs_n, COVs_n, TTL_n] = train_gmm(train_n, Ws_n, MUs_n, COVs_n);
    print('Iteration:', jj, ' Total log-likelihood:', TTL_t, 'for males;', TTL_n, 'for females')

P_t=0.5
P_n=1.0-P_t

score = []
for tst in test_t:
    ll_m = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
    ll_f = logpdf_gmm(tst, Ws_n, MUs_n, COVs_n)
    score.append((sum(ll_m) + np.log(P_t)) - (sum(ll_f) + np.log(P_n)))
print(score)
print('Fraction of correctly recognized targets: %f' % (np.mean(np.array(score) > 0)))

score = []
for tst in test_n:
    ll_m = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
    ll_f = logpdf_gmm(tst, Ws_n, MUs_n, COVs_n)
    score.append((sum(ll_m) + np.log(P_t)) - (sum(ll_f) + np.log(P_n)))
print(score)
print('Fraction of correctly recognized targets: %f' % (np.mean(np.array(score) < 0)))

# train_target = np.vstack(target_wav_train)
# train_non = np.vstack(non_target_wav_train)
# dim = train_target.shape[1]
#
# # PCA reduction to 2 dimensions
#
# cov_tot = np.cov(np.vstack([train_non, train_target]).T, bias=True)
# # take just 2 largest eigenvalues and corresponding eigenvectors
# d, e = scipy.linalg.eigh(cov_tot, eigvals=(dim - 2, dim - 1))
#
# train_n_pca = train_non.dot(e)
# train_t_pca = train_target.dot(e)
# plt.plot(train_n_pca[:, 1], train_n_pca[:, 0], 'b.', ms=1)
# plt.plot(train_t_pca[:, 1], train_t_pca[:, 0], 'r.', ms=1)
# plt.show()
#
# #print(non_target_wav_train)