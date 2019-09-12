import tensorflow as tf
import tensorflow.keras.layers as ly


def crnn_ref37(input_shape):
    # C'est le model qui va bien 40 log mel
    # epochcs = 100
    model = tf.keras.models.Sequential()
    model.add(ly.Conv2D(96, (5, 5), strides=(2, 2), activation='relu',
                        padding='same', input_shape=input_shape))
    model.add(ly.Dropout(0.25))
    model.add(ly.BatchNormalization())
    model.add(ly.MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(ly.Conv2D(96, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(ly.Dropout(0.25))
    model.add(ly.BatchNormalization())
    model.add(ly.MaxPool2D(pool_size=(1, 2), strides=(1, 1)))
    model.add(ly.Conv2D(96, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(ly.Dropout(0.25))
    model.add(ly.BatchNormalization())
    model.add(ly.MaxPool2D(pool_size=(1, 2), strides=(1, 1)))
    model.add(ly.Reshape((model.output_shape[1], model.output_shape[2] * model.output_shape[3])))
    model.add(ly.GRU(96, return_sequences=True, activation='tanh'))
    model.add(ly.Dropout(0.25))
    model.add(ly.BatchNormalization())
    model.add(ly.GRU(48, return_sequences=True, activation='tanh'))
    model.add(ly.Dropout(0.25))
    model.add(ly.BatchNormalization())
    model.add(ly.Flatten())
    return end_model(model)


def crnn_ref21(input_shape):
    # Ils utilise du raw audio, en entrée il doit y avoir 3200 amplitude
    model = tf.keras.models.Sequential()
    model.add(ly.Conv2D(64, (80, 80), strides=(1, 1), activation='relu',
                        padding='same', input_shape=input_shape))
    model.add(ly.BatchNormalization())
    model.add(ly.MaxPool2D(pool_size=(4, 1), strides=(1, 1)))
    model.add(ly.Conv2D(64,  kernel_size=4, activation='relu', strides=1, padding='same'))
    model.add(ly.BatchNormalization())
    model.add(ly.MaxPool2D(pool_size=(2, 1), strides=(1, 1)))
    model.add(ly.Conv2D(128, kernel_size=4, activation='relu', strides=1, padding='same'))
    model.add(ly.BatchNormalization())
    model.add(ly.MaxPool2D(pool_size=1, strides=1))
    model.add(ly.Reshape((model.output_shape[1], model.output_shape[2]*model.output_shape[3])))
    model.add(ly.GRU(128, return_sequences=True, activation='tanh', dropout=0.1))
    model.add(ly.GRU(128, return_sequences=True, activation='tanh', dropout=0.1))
    model.add(ly.Flatten())

    return end_model(model)


def dnn_ref20(input_shape):
    # Ce model est à essayer avec plusieurs sortes des parametres
    model = tf.keras.models.Sequential()
    model.add(ly.Dense(128, activation='relu', input_shape=input_shape))
    model.add(ly.BatchNormalization())
    model.add(ly.Dropout(0.2))
    model.add(ly.Dense(128, activation='relu'))
    model.add(ly.BatchNormalization())
    model.add(ly.Dropout(0.2))
    model.add(ly.Dense(128, activation='relu'))
    model.add(ly.BatchNormalization())
    model.add(ly.Dropout(0.2))
    model.add(ly.Dense(128, activation='relu'))
    model.add(ly.BatchNormalization())
    model.add(ly.Dropout(0.2))
    model.add(ly.Flatten())
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
