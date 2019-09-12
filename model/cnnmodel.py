import tensorflow as tf
import tensorflow.keras.layers as ly
"""
Les modeles à cnn
"""


def cnn_piczack(input_shape):
    # Pour ce model il doit y avoir 60 log mel spectogramme
    # 300 epochs avec learningrate = 0.002 ou 150 avec 0.01
    model = tf.keras.models.Sequential()
    model.add(ly.Conv2D(80, (57, 6), strides=(1, 1), activation='relu',
                        padding='same', input_shape=input_shape))
    model.add(ly.MaxPool2D(pool_size=(4, 3), strides=(1, 3)))
    model.add(ly.Conv2D(80, (1, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(ly.Dropout(0.5))
    model.add(ly.MaxPool2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(ly.Flatten())
    model.add(ly.Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(ly.Dropout(0.5))

    return end_model(model, optimizer='moment')


def cnn_ref2(input_shape):
    # Utiliser 64 mfcc avec les delts
    # 300 epochs avec learningrate = 0.002 ou 150 avec 0.01
    model = tf.keras.models.Sequential()
    model.add(ly.Conv2D(50, (60, 41), activation='relu', strides=(1, 1),
                        padding='same', input_shape=input_shape))
    model.add(ly.Conv2D(50, (3, 3), activation='relu', strides=(1, 1),
                        dilation_rate=2, padding='same'))
    model.add(ly.Conv2D(50, (3, 3), activation='relu', strides=(1, 1),
                        dilation_rate=2, padding='same'))
    model.add(ly.Conv2D(50, (3, 3), activation='relu', strides=(1, 1),
                        dilation_rate=2, padding='same'))
    model.add(ly.Dropout(0.5))
    model.add(ly.Flatten())
    model.add(ly.Dense(128, activation='relu'))

    return end_model(model, optimizer='moment')


def cnn_nous(input_shape):
    model = tf.keras.models.Sequential()
    model.add(ly.Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
                        padding='same', input_shape=input_shape))
    model.add(ly.Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(ly.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(ly.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(ly.MaxPool2D(2, 2))
    model.add(ly.Dropout(0.5))
    # added
    model.add(ly.Reshape((model.output_shape[1], model.output_shape[2] * model.output_shape[3])))
    model.add(ly.Bidirectional(ly.GRU(64, activation='tanh', return_sequences=True)))
    model.add(ly.Dense(64, activation='relu'))
    model.add(ly.Flatten())
    # End added

    # Previous
    # model.add(ly.Flatten())
    # model.add(ly.Dense(128, activation='relu'))
    # model.add(ly.Dense(64, activation='relu'))

    return end_model(model)


def cnn_ref19(input_shape):
    # Le model utilise la fusion tardive de deux model de CNN
    # Celui-ci est utilise le LMC: constitué des Log mel, chroma, spectral contrast et tonnetz
    # Le second est utilise le MF: MFCC, chroma, spectral contrast et tonnetz
    model = tf.keras.models.Sequential()
    model.add(ly.Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same',
                        input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(ly.BatchNormalization())
    model.add(ly.Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(ly.BatchNormalization())
    model.add(ly.Dropout(0.5))
    model.add(ly.MaxPool2D(2, 2))
    model.add(ly.Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(ly.BatchNormalization())
    model.add(ly.Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(ly.BatchNormalization())
    model.add(ly.Dropout(0.5))
    model.add(ly.Flatten())
    model.add(ly.Dense(1024, activation='sigmoid'))
    model.add(ly.Dropout(0.5))

    return end_model(model)


def end_model(model, optimizer='adam'):
    model.add(ly.Dense(8, activation='softmax'))
    model.summary()
    if optimizer is 'adam':
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    elif optimizer is 'moment':
        model.compile(optimizer=tf.train.MomentumOptimizer(0.002, 0.9),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    return model
