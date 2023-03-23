import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras


def normalizeImg(image, label):
    return tf.cast(image, tf.float32) / 255., label


def main():
    # print(ds)
    [trainDs, testDs] = tfds.load(
        "beans", 
        split=['train', "test"], 
        shuffle_files=True, 
        as_supervised=True,
    )
    trainDs = trainDs.map(normalizeImg, num_parallel_calls=tf.data.AUTOTUNE)
    trainDs = trainDs.prefetch(tf.data.AUTOTUNE)
    trainDs = trainDs.batch(8)
    trainDs = trainDs.cache()
    testDs = testDs.map(normalizeImg, num_parallel_calls=tf.data.AUTOTUNE)
    testDs = testDs.prefetch(tf.data.AUTOTUNE)
    testDs = testDs.batch(8)
    testDs = testDs.cache()

    imgSize = 180
    resizeAndRescale = keras.Sequential([
        keras.layers.Resizing(imgSize, imgSize),
        # keras.layers.Rescaling(1./255),
    ])
    
    dataAugmentation = tf.keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
    ])

    model = keras.Sequential([
        resizeAndRescale,
        dataAugmentation,
        keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    # model.summary()
    model.fit(
        trainDs, 
        epochs=10,
        validation_data=testDs,
    )


main()
