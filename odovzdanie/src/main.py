import os.path
import re
from os import path

import pandas as pd
import tensorflow as tf
from keras.models import Sequential

import img_class_nn
import audio_GMM
from projekt_lib import eval_load

if __name__ == "__main__":

    model = Sequential()
    if path.isdir('my_model'):
        model = tf.keras.models.load_model("my_model")

    else:
        model = img_class_nn.train_model()

    score, _ = audio_GMM.GMM_decision('train_data/non_target_train', 'train_data/target_train', 'eval')

    eval_data, files = eval_load('eval')
    est = model.predict(eval_data)
    df = pd.DataFrame(est)
    df['filename'] = files
    df['wav_score'] = score

    if os.path.isfile('results.txt'):
        os.remove('results.txt')

    png_threshold = 0.9
    wav_threshold = 0.98

    f = open("results.txt", "a")
    for num, row in df.iterrows():
        filename = re.search("eval_[0-9][0-9][0-9][0-9]", row['filename'])
        filename = row['filename'][filename.start():filename.end()]
        combined_score = row.iloc[1] * row['wav_score']

        if combined_score > (png_threshold * wav_threshold):
            hard_decision = 1
        else:
            hard_decision = 0

        string = "{} {} {}\n".format(filename, round(combined_score, 2), hard_decision)
        f.write(string)

    f.close()
