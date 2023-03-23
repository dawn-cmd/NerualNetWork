import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def main():
    BATCH_SIZE = 256
    VOCAB_SIZE= 10000
    MAX_SEQ_LEN = 150
    [trainDs, valDs] = tfds.load(
        "imdb_reviews",
        split=["train[:90%]", "train[90%:]"],
        as_supervised=True,
        batch_size=BATCH_SIZE
    )
    tokenize = Tokenizer(VOCAB_SIZE, oov_token="<OOV>")

    def prepSentences(s, l, tokenizer: Tokenizer, fit: bool, maxlen: int):
        if fit:
            tokenizer.fit_on_texts(s)
        s = tokenizer.texts_to_sequences(s)
        s = pad_sequences(s, maxlen=maxlen, truncating="post")
        s = tf.expand_dims(s, -1)
        l = np.array(l)
        return s, l, tokenizer
    
    def getTxtAndLb(Ds: tf.data.Dataset):
        txt = []
        lb = []
        for element in Ds.as_numpy_iterator():
            for sentence in element[0]:
                txt.append(bytes.decode(sentence))
            for label in element[1]:
                lb.append(label)
        return txt, lb
    
    trainTxt, trainLb = getTxtAndLb(trainDs)
    trainTxt, trainLb, tokenize = prepSentences(trainTxt, trainLb, tokenize, fit=True, maxlen=MAX_SEQ_LEN)
    valTxt, valLb = getTxtAndLb(valDs)
    valTxt, valLb, tokenize = prepSentences(valTxt, valLb, tokenize, fit=False, maxlen=MAX_SEQ_LEN)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_SEQ_LEN),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(64, 8, activation=tf.nn.relu),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.LSTM(32, return_sequences=True),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    h = model.fit(
        trainTxt, trainLb,
        epochs=10,
        validation_data=(valTxt, valLb),
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
