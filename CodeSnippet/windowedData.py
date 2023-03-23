import numpy as np
import tensorflow as tf


def windowedData(X: np.array, winSize: int, batchSize: int, shuffleBuffer: int):
    Ds = tf.data.Dataset.from_tensor_slices(X)
    Ds = Ds.window(winSize + 1, shift=1, drop_remainder=True)
    Ds = Ds.flat_map(lambda window: window.batch(winSize + 1))
    Ds = Ds.map(lambda win: (win[:-1], win[-1]))
    Ds = Ds.shuffle(shuffleBuffer)
    Ds = Ds.batch(batch_size=batchSize).prefetch(1)
    return Ds

