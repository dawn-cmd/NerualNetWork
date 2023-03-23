import tensorflow as tf


def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1) if epoch > 5 else lr


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)