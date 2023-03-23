import tensorflow as tf
from tensorflow import keras
import numpy as np

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print('\nTrain ends at 99%')
            self.model.stop_training = True
        return super().on_epoch_end(epoch, logs)

def main():
    (trainingImage, trainingLabel), (testImage, testLabel) = keras.datasets.mnist.load_data()
    print(len(trainingImage))

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    model.fit(trainingImage, trainingLabel, epochs=20, callbacks=[myCallback()])
    model.evaluate(testImage, testLabel)

main()
