import keras.metrics
import tensorflow as tf
import tensorflow_datasets as tfds


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


def main():
    [trainDs, testDS] = tfds.load(
        "cassava",
        split=["train", "test"],
        as_supervised=True,
        shuffle_files=True,
    )
    
    trainDs = prepImg(trainDs, 256, 16, shuffle=True, aug=True)
    testDS = prepImg(testDS, 256, 16, shuffle=True, aug=False)
    
    model = tf.keras.models.Sequential([
        # tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        # tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-6),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.Accuracy],
    )
    model.fit(
        trainDs,
        epochs=15,
        validation_data=testDS
    )
    
    
main()
