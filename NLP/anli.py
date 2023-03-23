import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1) if epoch > 5 else lr


def prepSentences(s, l, tokenizer: Tokenizer, fit: bool, maxlen: int):
    if fit:
        tokenizer.fit_on_texts(s)
    s = tokenizer.texts_to_sequences(s)
    s = pad_sequences(s, maxlen=maxlen, truncating="post")
    l = np.array(l)
    return s, l, tokenizer


def printResult(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric], loc="lower right")
    plt.show()


def main():
    maxlen = 100
    vocabSize = 10000
    [trainDs, testDs] = tfds.load(
        "anli/r3",
        split=["train[:80%]", "train[80%:]"],
        # shuffle_files=True,
    )
    # trainDs = trainDs.shuffle(1000)
    # testDs = testDs.shuffle(1000)
    trainSentences = []
    trainLabels = []
    tokenizer = Tokenizer(vocabSize, oov_token="<OOV>")
    for element in trainDs.as_numpy_iterator():
        # print(element)
        trainSentences.append(bytes.decode(element["context"]).lower())
        trainLabels.append(element["label"])
    trainSentences, trainLabels, tokenizer = prepSentences(trainSentences, trainLabels, tokenizer, True, maxlen)
    testSentences = []
    testLabels = []
    for element in testDs.as_numpy_iterator():
        testSentences.append(bytes.decode(element["context"]).lower())
        testLabels.append(element["label"])
    testSentences, testLabels, tokenizer = prepSentences(testSentences, testLabels, tokenizer, False, maxlen)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocabSize, 512, input_length=maxlen),
        tf.keras.layers.LSTM(16, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.00001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    h = model.fit(
        trainSentences,
        trainLabels,
        epochs=30,
        validation_data=(testSentences, testLabels),
        batch_size=512,
        # callbacks=[callback]
    )
    printResult(h, "accuracy")

    
main()