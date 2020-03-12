import pickle
import numpy


def load_word_embeddings(path):
    words, embeddings = pickle.load(open(path, 'rb'), encoding='latin1')
    print("Emebddings shape is {}".format(embeddings.shape))
    return words, embeddings


def load_treebanks(path):
    data = None
    with open(path, 'r') as f:
        data = f.read().split('\n')

    return data


def train_test_split(data, train_size, dev_size, test_size):
    n = len(data)
    n_train = int(n*train_size )
    n_dev = int(n*(train_size + dev_size))

    train_data = data[:n_train]
    dev_data = data[n_train:n_dev]
    test_data = data[n_dev:]

    return train_data, dev_data, test_data
