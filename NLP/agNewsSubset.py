from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def main():
    # import data from ag_news_subset
    [trainDs, testDs], info = tfds.load(
        "ag_news_subset",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    
    # process train dataset
    trainSentences = []
    trainLabels = []
    for element in trainDs.as_numpy_iterator():
        trainSentences.append(bytes.decode(element[0]))
        trainLabels.append(element[1])
    tokenize = Tokenizer(num_words=1e4)
    tokenize.fit_on_texts(trainSentences)
    trainSentences = tokenize.texts_to_sequences(trainSentences)
    trainPadded = pad_sequences(trainSentences, maxlen=120, truncating="post")
    trainLabels = np.array(trainLabels)

    # process test dataset
    testSentences = []
    testLabels = []
    for element in testDs.as_numpy_iterator():
        testSentences.append(bytes.decode(element[0]))
        testLabels.append(element[1])
    testSentences = tokenize.texts_to_sequences(testSentences)
    testPadded = pad_sequences(testSentences, maxlen=120, truncating="post")
    testLabels = np.array(testLabels)
    
    model = keras.models.Sequential([
        keras.layers.Embedding(int(1e4), 16, input_length=120),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(4, activation="softmax"),
    ])
    # model.summary()
    # print(tensorflow.__version__)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"],
                  optimizer=keras.optimizers.Adam(0.001))
    model.fit(trainPadded, trainLabels, validation_data=(testPadded, testLabels), epochs=10)
    

if __name__ == '__main__':
    main()
