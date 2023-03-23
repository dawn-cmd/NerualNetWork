import tensorflow as tf
from tensorflow import keras


def main():
    win_size = 5
    ds = tf.data.Dataset.range(10)
    ds = ds.window(win_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda win: win.batch(win_size + 1))
    ds = ds.map(lambda win: (win[:-1], win[-1]))
    ds = ds.shuffle(buffer_size=10)
    ds = ds.batch(2).prefetch(1)
    
    model = keras.models.Sequential([
        keras.layers.Dense(16, input_shape=[win_size], activation="relu"),
        keras.layers.Dense(4, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(
        loss="mae",
        metrics=["accuracy"],
        optimizer=keras.optimizers.SGD(1e-6)
    )
    model.fit(ds, epochs=1000)
    
    
main()
