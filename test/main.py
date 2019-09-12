"""
fichier de base pour le lacement des test
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow.keras as keras
import numpy as np
import model.cnnmodel as cnn
import model.dnnmodel as dnn
import model.rnnmodel as rnn
import dataset.load_dataset as ds


BATCH_SIZE = 32
EPOCHS = 20
shuffle = True

# *************************  CONVOLUTIONS ***************************************
X, y = ds.load_default()
input_shape = (X.shape[1], X.shape[2], 1)  # for cnn
# print('**************************   Piczack net stats  ************************************')
# model = cnn.cnn_piczack(input_shape)
# model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=shuffle)
# print('**************************     REF2 net stats   ************************************')
# model = cnn.cnn_ref2(input_shape)
# model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=shuffle)
# model = cnn.cnn_git(input_shape)
# model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=shuffle)

# *************************  RECURRENT  ***************************************
# X, y = ds.load_default(mode='time')
# input_shape = (X.shape[1], X.shape[2])  # for rnn
# print('**************************   GIT VIDEO net stats  ************************************')
# model = rnn.rnn_ref1(input_shape)
# model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=shuffle)
# print('**************************     REF47 net stats   ************************************')
# model = rnn.blstm_ref47(input_shape)
# model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=shuffle)
# print('**************************     REF3 net stats   ************************************')
model = rnn.rnn_ref3(input_shape)
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=shuffle)

# Sauvegarde du model complet
# model.save('crnn3.h5')

# Restore model : il faut tenir compte du input shape
model = keras.models.load_model('crnn2.h5')
model.summary()
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""
o le model cnn_git a eu 81.93
o le model cnn_piczack a eu 66.4 
o le model cnn_ref2 atteint 67.0 
o le model rnn_git atteint 62.33 
o le model rnn_ref1 est exclus de la competition car a la plus faible accuracy soit 58.74% et très difficile
    à trainer en raisons de ses 1964552 Params dont 1962504 train-Params
o le model blstm_ref47 84.22 
o le model rnn_ref3 atteint 82.12 on peut changer les carcteristique ou augmenter les epochs
o le model cnn_ref19 a eu 68.63
o le model crnn_ref37 a eu 60.98
o le model dnn_ref20 a eu 75.77
o le model crnn_ref21 a eu 72.51

**********************************   Pour les 20 mels spectogramme   ******************************
o le model crnn_ref21 a eu 69.34
o le model dnn_ref20 a eu 50.76
o le model cnn_ref19 a eu 45.82
o le model cnn_git a eu 65.35
o le model cnn_ref2 a eu 26.88
o le model blstm_ref47 a eu 74.49 avec deltas 13.40
o le model rnn_ref3 a eu 61.02

**********************************   les gamatones   ******************************
o le model blstm_ref47 a eu 79.31 (64) 73.41(20) 76.30 (40)
o le model rnn_ref3 a eu 77.44
o le model cnn_git a eu 88.44
o le model crnn_ref21 a eu 83.14
o le model cnn_ref19 a eu 75.3
o le model cnn_ref2 a eu 74.22

**********************************   les MFCC+chroma+spec+tonetz   ******************************
o le model cnn_ref19 a eu 72.06
o le model cnn_git a eu 86.77
o le model rnn_ref3 a eu 83.86
o le model blstm_ref47 a eu 81.59 il y a eu stagnation pas besoin d'augmenter l'epoch

**********************************   les gama+chroma+spec+tonetz   ******************************
o  le model rnn_ref3 environ 50.xx au 5 epoch
o cnn_git 55.76
o blstm_ref47 50.57

*******************************   logmel+chroma+spec+tonetz  ***********************************
o cnn_git 69.32
o cnn_git 68.75
o blst47 a la carte


o CRNN ours with les MFCC+chroma+spec+tonetz 90.53
o CRNN ours with gamatones 91.25
"""

