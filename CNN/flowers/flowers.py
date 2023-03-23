import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import matplotlib.pyplot as plt


def main():
    BATCH_SIZE = 16
    IMG_SIZE = 120
    trainDs = tf.keras.utils.image_dataset_from_directory(
        "./flower_photos",
        validation_split=0.2,
        subset="training",
        seed=123,
        batch_size=BATCH_SIZE,
    )
    valDs = tf.keras.utils.image_dataset_from_directory(
        "./flower_photos",
        validation_split=0.2,
        subset="validation",
        seed=123,
        batch_size=BATCH_SIZE,
    )

    def prepImg(dataset, imgSize: int, batchSize: int, shuffle: bool, aug: bool):
        resizeAndRescale = tf.keras.models.Sequential([
            tf.keras.layers.Resizing(imgSize, imgSize),
            tf.keras.layers.Rescaling(1. / 255)
        ])
        dataAugment = tf.keras.models.Sequential([
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(0.5),
            tf.keras.layers.RandomZoom(0.5),
        ])
        dataset = dataset.map(
            lambda x, y: (resizeAndRescale(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        if aug:
            dataset = dataset.map(
                lambda x, y: (dataAugment(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        # dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    trainDs = prepImg(trainDs, imgSize=IMG_SIZE, batchSize=BATCH_SIZE, shuffle=False, aug=True)
    valDs = prepImg(valDs, imgSize=IMG_SIZE, batchSize=BATCH_SIZE, shuffle=False, aug=False)
    model = tf.keras.models.Sequential([
        Conv2D(16, (3, 3), padding="same", activation=tf.nn.relu),
        MaxPooling2D(),
        # Dropout(0.2),
        Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu),
        MaxPooling2D(),
        # Dropout(0.2),
        Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation=tf.nn.relu),
        # Dense(64, activation=tf.nn.relu),
        Dense(5, activation=tf.nn.softmax)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    h = model.fit(
        trainDs,
        epochs=50,
        validation_data=valDs,
    )

    def printResult(history, metric):
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric], '')
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([metric, 'val_' + metric], loc="lower right")
        plt.show()
    
    printResult(h, "accuracy")
    

main()
