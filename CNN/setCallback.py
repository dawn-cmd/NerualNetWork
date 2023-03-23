import tensorflow as tf
from tensorflow import keras
import numpy as np

class myCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.8:
            print('\nCancel train at 80%')
            self.model.stop_training = True
        return super().on_epoch_end(epoch, logs)

def main():
    fashionMnist = keras.datasets.fashion_mnist
    (trainImage, trainLabel), (testImage, testLabel) = fashionMnist.load_data()
    # print('load data over')
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    callback = myCallBack()
    model.fit(trainImage, trainLabel, epochs=20, callbacks=[callback])

main()
