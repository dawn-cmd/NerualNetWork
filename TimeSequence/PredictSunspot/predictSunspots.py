import csv

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda


def main():
    BATCH_SIZE = 128
    WIN_SIZE = 30
    SHUFFLE_BUFFER = 1000
    timeStep = []
    series = []
    
    with open("./Sunspots.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            timeStep.append(int(row[0]))
            series.append(float(row[2]))
            
    timeStep = np.array(timeStep)
    series = np.array(series)
    trainTime, valTime = timeStep[:int(len(timeStep) * 0.8)], timeStep[int(len(timeStep) * 0.8):]
    trainX, valX = series[:int(len(series) * 0.8)], series[int(len(series) * 0.8):]
    
    def windowedData(X: np.array, winSize: int, batchSize: int, shuffleBuffer: int):
        Ds = tf.data.Dataset.from_tensor_slices(X)
        Ds = Ds.window(winSize + 1, shift=1, drop_remainder=True)
        Ds = Ds.flat_map(lambda window: window.batch(winSize + 1))
        Ds = Ds.map(lambda win: (win[:-1], win[1:]))
        Ds = Ds.shuffle(shuffleBuffer)
        Ds = Ds.batch(batch_size=batchSize).prefetch(1)
        return Ds
    
    trainDs = windowedData(trainX, WIN_SIZE, BATCH_SIZE, SHUFFLE_BUFFER)
    valDs = windowedData(valX, WIN_SIZE, BATCH_SIZE, SHUFFLE_BUFFER)
    
    model = tf.keras.models.Sequential([
        Conv1D(64, 3, input_shape=[WIN_SIZE, 1], activation=tf.nn.relu),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(128, activation=tf.nn.relu),
        Dense(64, activation=tf.nn.relu),
        Dense(1),
        Lambda(lambda x: x * 400)
    ])
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10 ** (epoch / 20)
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=8e-7, momentum=0.9),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.mae]
    )
    h = model.fit(
        trainDs,
        epochs=100,
        # callbacks=[lr_schedule],
        validation_data=valDs
    )

    # Define the learning rate array
    lrs = 1e-8 * (10 ** (np.arange(100) / 20))

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Set the grid
    plt.grid(True)

    # Plot the loss in log scale
    plt.semilogx(lrs, h.history["loss"])

    # Increase the tickmarks size
    plt.tick_params('both', length=10, width=1, which='both')

    # Set the plot boundaries
    plt.axis([1e-8, 1e-3, 0, 100])
    plt.show()
    

main()
    