import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import dataset.load_dataset as ds
import seaborn as sns

"""
Sert à generer la matrice de confusion
"""

X, y = ds.load_default()
model = keras.models.load_model('crnn2.h5')
# model.summary()
y_pred =model.predict_classes(X)
classes= ds.get_class()

con_mat = tf.math.confusion_matrix(labels=y, predictions=y_pred)

with tf.Session() as sess:
    con_mat = sess.run(con_mat)

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1), decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()