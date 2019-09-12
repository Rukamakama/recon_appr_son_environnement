import tensorflow as tf
import tensorflow.keras.layers as ly
"""
Les modeles à rnn
"""


def rnn_git(input_shape):
    # shape of our dat for RNN is (n, time, feat)
    model = tf.keras.models.Sequential()
    model.add(ly.LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(ly.LSTM(128, return_sequences=True))
    model.add(ly.Dropout(0.5))
    model.add(ly.TimeDistributed(ly.Dense(64, activation='relu')))
    model.add(ly.TimeDistributed(ly.Dense(32, activation='relu')))
    model.add(ly.TimeDistributed(ly.Dense(16, activation='relu')))
    model.add(ly.TimeDistributed(ly.Dense(8, activation='relu')))
    model.add(ly.Flatten())

    return end_model(model)


def blstm_ref47(input_shape):
    # C'est un model qui utilise 40 log mel
    # epochs = 20 minibatch = 200
    model = tf.keras.models.Sequential()
    model.add(ly.Bidirectional(ly.LSTM(100, return_sequences=True, dropout=0.25), input_shape=input_shape))
    model.add(ly.Bidirectional(ly.LSTM(100, activation='tanh',  return_sequences=True)))
    model.add(ly.Bidirectional(ly.LSTM(100, activation='tanh', return_sequences=True)))
    model.add(ly.Bidirectional(ly.LSTM(100, activation='tanh', return_sequences=True)))
    model.add(ly.BatchNormalization())
    model.add(ly.Flatten())

    return end_model(model, optimizer='RMSprop')


def rnn_ref3(input_shape):
    # C'est le model qui donne de bons resultats avec les rnn, ils ont fait recourt à late fusion
    # donc nous allons prendre le performance un à un: 1. 64 gamatones,2. 60 MFCC,3. 20log mel avec leurs
    #  derivées de premier et second ordre, ZCR, centroides, pectral bandwith, 4subbande energy
    # Le model contient peut des parametres et il faudra penser à augmenter le nombre d'epoque
    model = tf.keras.models.Sequential()
    model.add(ly.GRU(256, return_sequences=True, dropout=0.1, input_shape=input_shape))
    model.add(ly.GRU(256, return_sequences=True, dropout=0.1, activation='tanh'))
    model.add(ly.Flatten())
    return end_model(model)


def rnn_ref1(input_shape):
    # C'est le model qui va bien avec les MFCC, ils ont un tableau des parmetres d'entre
    # de 20 MFCC plus leurs derivees de premiers et seconds ordre, ce qui fait 60
    model = tf.keras.models.Sequential()
    model.add(ly.LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(ly.TimeDistributed(ly.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))))
    model.add(ly.Dropout(0.25))
    model.add(ly.BatchNormalization())
    model.add(ly.TimeDistributed(ly.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))))
    model.add(ly.Dropout(0.25))
    model.add(ly.BatchNormalization())
    model.add(ly.TimeDistributed(ly.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))))
    model.add(ly.Dropout(0.25))
    model.add(ly.BatchNormalization())
    model.add(ly.TimeDistributed(ly.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))))
    model.add(ly.Dropout(0.25))
    model.add(ly.BatchNormalization())
    model.add(ly.Bidirectional(ly.GRU(256, activation='relu', return_sequences=True, dropout=0.25,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.1))))
    model.add(ly.Bidirectional(ly.GRU(256, activation='relu', return_sequences=True, dropout=0.25,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.1))))
    model.add(ly.BatchNormalization())
    model.add(ly.Flatten())

    return end_model(model)


def end_model(model, optimizer='adam'):
    model.add(ly.Dense(8, activation='softmax'))
    model.summary()
    if optimizer is 'moment':
        model.compile(optimizer=tf.train.MomentumOptimizer(0.002, 0.9),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimizer is 'RMSprop':
        model.compile(optimizer=tf.train.RMSPropOptimizer(0.005, decay=0.9),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
