import tensorflow_datasets
import tensorflow as tf
import numpy as np


def main():
    [trainDs, testDs] = tensorflow_datasets.load(
        "caltech101",
        split=["test", "train"],
        as_supervised=True,
        shuffle_files=True,
    )
    
    resizeAndRescale = tf.keras.models.Sequential([
        tf.keras.layers.Resizing(100, 100),
        tf.keras.layers.Rescaling(1. / 255),
    ])
    
    trainDs = trainDs.map(lambda x, y: (resizeAndRescale(x), y))
    testDs = testDs.map(lambda x, y: (resizeAndRescale(x), y))
    trainDs = trainDs.prefetch(tf.data.AUTOTUNE)
    testDs = testDs.prefetch(tf.data.AUTOTUNE)
    trainDs = trainDs.cache()
    testDs = testDs.cache()
    trainDs = trainDs.batch(8)
    testDs = testDs.batch(8)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.5),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(102, activation=tf.nn.softmax),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(
        trainDs,
        epochs=20,
        validation_data=testDs,
        batch_size=16
    )


main()
