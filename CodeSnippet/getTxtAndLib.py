import tensorflow as tf


def getTxtAndLb(Ds: tf.data.Dataset):
    txt = []
    lb = []
    for element in Ds.as_numpy_iterator():
        for sentence in element[0]:
            txt.append(bytes.decode(sentence))
        for label in element[1]:
            lb.append(label)
    return txt, lb