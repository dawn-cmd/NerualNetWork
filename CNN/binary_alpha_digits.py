import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def plot_loss_acc(history):
  '''Plots the training and validation loss and accuracy from a history object'''
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(acc))

  plt.plot(epochs, acc, 'bo', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training Loss')
  plt.plot(epochs, val_loss, 'b', label='Validation Loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()

def main():
    dsTrain, dsTest = tfds.load(
        "binary_alpha_digits", 
        split=["train[:90%]", "train[90%:]"], 
        shuffle_files=True, 
        as_supervised=True,
    )
    dsTrain = dsTrain.batch(64)
    dsTest = dsTest.batch(8)
    
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(20, 16, 1)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(36, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    model.fit(
        dsTrain,
        epochs=200,
        validation_data=dsTest, 
    )

main()
