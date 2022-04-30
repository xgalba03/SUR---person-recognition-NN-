import os

import re
import numpy as np
import pandas as pd
from numpy.random import randint
from projekt_lib import train_gmm, logpdf_gmm, process_audio


def GMM_decision(train_data_nontarget, train_data_target, test_data):
    print("processing traning audio data")
    train_n, _ = process_audio(train_data_nontarget)
    train_t, _ = process_audio(train_data_target)
    print("processing test audio data")
    test, files = process_audio(test_data)

    train_n = train_n.values()
    train_t = train_t.values()
    test = test.values()

    train_n = np.vstack(train_n)
    train_t = np.vstack(train_t)

    M_t = 3
    MUs_t = train_t[randint(1, len(train_t), M_t)]
    COVs_t = [np.cov(train_t.T)] * M_t
    Ws_t = np.ones(M_t) / M_t

    M_n = 10
    MUs_n = train_n[randint(1, len(train_n), M_n)]
    COVs_n = [np.cov(train_n.T)] * M_n
    Ws_n = np.ones(M_n) / M_n

    print("iterating EM algorithm")
    for jj in range(30):
        [Ws_t, MUs_t, COVs_t] = train_gmm(train_t, Ws_t, MUs_t, COVs_t)
        [Ws_n, MUs_n, COVs_n] = train_gmm(train_n, Ws_n, MUs_n, COVs_n)

    aprior_p = 0.5

    score = []
    min = 1000
    max = -1000
    print("evaluating tests")
    for tst in test:
        ll_m = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
        ll_f = logpdf_gmm(tst, Ws_n, MUs_n, COVs_n)
        log_score = (sum(ll_m) + np.log(aprior_p)) - (sum(ll_f) + np.log(aprior_p))
        score.append(log_score)
        if (log_score < min):
            min = log_score
        elif (log_score > max):
            max = log_score

    percentage = []
    for value in score:
        percentage.append((((value - min) / (max - min)) * (100 - 0) + 0) / 100)

    return percentage, files


if __name__ == "__main__":

    score, files = GMM_decision('train_data/non_target_train', 'train_data/target_train', 'eval')
    textfile = open("percentage.txt", "w")
    for element in score:
        textfile.write(f"{element}\n")
    textfile.close()

    if os.path.isfile('audio_GMM.txt'):
        os.remove('audio_GMM.txt')

    df = pd.DataFrame(score)
    df['filename'] = files

    f = open("audio_GMM.txt", "a")
    for num, row in df.iterrows():
        filename = re.search("eval_[0-9][0-9][0-9][0-9]", row['filename'])
        filename = row['filename'][filename.start():filename.end()]

        if float(row.iloc[0]) > 0.98:
            hard_decision = 1
        else:
            hard_decision = 0

        string = "{} {} {}\n".format(filename, round(row.iloc[0], 2), hard_decision)
        f.write(string)

    f.close()

