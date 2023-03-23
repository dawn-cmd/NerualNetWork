import tensorflow as tf


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

