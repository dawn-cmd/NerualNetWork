from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


def prepSentences(s, l, tokenizer: Tokenizer, fit: bool, maxlen: int):
    if fit:
        tokenizer.fit_on_texts(s)
    s = tokenizer.texts_to_sequences(s)
    s = pad_sequences(s, maxlen=maxlen, truncating="post")
    l = np.array(l)
    return s, l, tokenizer