"""
Pour la lecture des données depuis les fichiers
Le pretraitement se fait ici. Le données sont adaptés au model
"""

import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from dataset.gtg import gammatonegram as gamatones

# csv_path = '../dataset/ESC-SEC/esc_sec.csv'
# raw_path = '../dataset/ESC-SEC/audio/'
csv_path = '../dataset/ESC-SEC/trials.csv'
raw_path = '../dataset/ESC-SEC/trials/'


def load_default(mode='conv'):
    config = Config(mode=mode)
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=proba_dist)
        file = np.random.choice(df[df.label == rand_class].index)
        ogg, rate = librosa.load(raw_path + file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, ogg.shape[0] - config.step)
        sample = ogg[rand_index:rand_index + config.step]

        # Loading raw data
        # X_sample = sample.T

        # Loading 20 mfcc
        # mfcc = librosa.feature.mfcc(y=sample, sr=rate, n_mfcc=config.nfeat).T

        # Load mel spectograms 10 mel with 5 delat1 and 5 deltat2
        # mels = librosa.feature.melspectrogram(y=sample, sr=rate, n_mels=config.mel).T
        # delta1 = librosa.feature.delta(mels, width=config.delta, order=1)
        # delta2 = librosa.feature.delta(mels, width=config.delta, order=2)
        # features = mels
        # features = np.append(features, delta1, axis=1)
        # features = np.append(features, delta2, axis=1)

        # Loading 20 gamatones
        features, freq = gamatones(sample, sr=config.rate, fmin=20, fmax=int(rate/2.), N=40)

        # Loading 5 chroma 7 spec and 6 tonnez = 18 caracteristiques
        # chroma = librosa.feature.chroma_cqt(y=sample, sr=rate, n_chroma=5).T
        # tonnetz = librosa.feature.tonnetz(y=sample, sr=rate, chroma=chroma).T
        # spec_contr = librosa.feature.spectral_contrast(y=sample, sr=rate).T

        # for dealing with gamatones
        # features = np.delete(gamaton, [5, 6, 7], axis=1).T

        # features = np.append(mfcc, chroma, axis=1)
        # features = np.append(features, tonnetz, axis=1)
        # features = np.append(features, spec_contr, axis=1)

        _min = min(np.amin(features), _min)
        _max = max(np.amax(features), _max)
        X.append(features if config.mode == 'conv' else features.T)
        y.append(classes.index(label))
    X = np.array(X)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=8)

    return X, y


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
for f in df.index:
    y, sr = librosa.load(raw_path + f)
    df.at[f, 'length'] = y.shape[0]/sr

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum() / 0.1)
proba_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=proba_dist)


def get_class():
    return classes