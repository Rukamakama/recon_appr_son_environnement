""""
Les predictions se font ici
"""

import tensorflow.keras as keras
import pandas as pd
import numpy as np
import librosa
from dataset.gtg import gammatonegram as gamatones

csv_path = '../dataset/ESC-SEC/esc_sec.csv'
raw_path = '../dataset/ESC-SEC/audio/'


def predict(sample):
    model = keras.models.load_model('../test/crnn1.h5')
    X = extraire_caract(sample)
    y_prob = model.predict(X)
    index = int(np.argmax(y_prob))
    return classes[index]


def extraire_caract(sample):
    config = Config()
    X = []
    _min, _max = float('inf'), -float('inf')
    rand_index = np.random.randint(0, sample.shape[0] - config.step)
    sample = sample[rand_index:rand_index + config.step]
    sample = sample.reshape((2205, ))
    # features, freq = gamatones(sample, sr=config.rate, fmin=20, fmax=int(config.rate / 2.), N=40)

    rate = config.rate
    mfcc = librosa.feature.mfcc(y=sample, sr=rate, n_mfcc=config.nfeat).T
    chroma = librosa.feature.chroma_cqt(y=sample, sr=rate, n_chroma=5).T
    tonnetz = librosa.feature.tonnetz(y=sample, sr=rate, chroma=chroma).T
    spec_contr = librosa.feature.spectral_contrast(y=sample, sr=rate).T
    features = np.append(mfcc, chroma, axis=1)
    features = np.append(features, tonnetz, axis=1)
    features = np.append(features, spec_contr, axis=1)

    _min = min(np.amin(features), _min)
    _max = max(np.amax(features), _max)
    X.append(features)

    X = np.array(X)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    return X

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=20, nfft=512, rate=22050, delta=5, mel=20):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.delta = delta
        self.mel = mel
        self.rate = rate
        self.step = int(rate / 10)


df = pd.read_csv(csv_path)  # le data frame
df.set_index('fname', inplace=True)

classes = list(np.unique(df.label))
for classe in classes:
    print(classe)




