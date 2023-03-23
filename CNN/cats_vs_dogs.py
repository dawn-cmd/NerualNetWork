import tensorflow as tf
import tensorflow_datasets
import matplotlib.pyplot as plt


def prepImg(dataset, imgSize: int, batchSize: int, shuffle: bool, aug: bool):
    resizeAndRescale = tf.keras.models.Sequential([
        tf.keras.layers.Resizing(imgSize, imgSize),
        tf.keras.layers.Rescaling(1. / 255)
    ])
    dataAugment = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomZoom(0.5),
        tf.keras.layers.RandomRotation(0.5),
    ])
    dataset = dataset.map(
        lambda x, y: (resizeAndRescale(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batchSize)
    if aug:
        dataset = dataset.map(
            lambda x, y: (dataAugment(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1) if epoch > 5 else lr


def printResult(history):
    plt.plot(history.history["accuracy"], label='accuracy')
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    plt.show()


def main():
    [trainDs, testDs] = tensorflow_datasets.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True,
        shuffle_files=True,
    )
    
    trainDs = prepImg(trainDs, 64, 128, shuffle=True, aug=True)
    testDs = prepImg(testDs, 64, 128, shuffle=True, aug=False)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(
        trainDs,
        epochs=20,
        validation_data=testDs,
        # callbacks=[callback],
    )
    printResult(history)


main()
