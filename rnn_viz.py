import argparse
import pickle
from os import path

import pandas as pd
import numpy as np
import tensorflow as tf
np.random.seed(123)
tf.set_random_seed(123)

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn.metrics import confusion_matrix
from utils import get_train_test_split, plot_confusion_matrix
from rnn_model import full_model
from rnn_constants import MAXLEN, RARITIES, FULL_INPUTS, \
    DEFAULT_WEIGHTS, DEFAULT_EMBEDDING, DEFAULT_TOKENIZER

def visualize(m_test, x_test, y_test, model, variant=None):
    # viz accuracy
    print('predicting test set...')
    y_pred = model.predict([m_test, x_test])
    conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    name = 'RNN'
    if variant is not None:
        name += '_' + variant
    plot_confusion_matrix(conf_mat, RARITIES, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model results')
    parser.add_argument('-model', '-m', help='model variant',
                        default='lstm')
    parser.add_argument('-weights', '-w', help='the weights to use',
                        default=DEFAULT_WEIGHTS)
    parser.add_argument('-embedding', '-e', help='the word_embedding to use',
                        default=DEFAULT_EMBEDDING)
    parser.add_argument('-tokenizer', '-t', help='the weights to use',
                        default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    kw = vars(args)

    # check file existence before we load everything into memory
    weights_fname = kw['weights']
    if not path.isfile(weights_fname):
        print('weights file "%s" does not exist' % weights_fname)
        exit()

    embedding_fname = kw['embedding']
    if not path.isfile(embedding_fname):
        print('embedding file "%s" does not exist' % embedding_fname)
        exit()

    tokenizer_fname = kw['tokenizer']
    if not path.isfile(tokenizer_fname):
        print('tokenizer file "%s" does not exist' % tokenizer_fname)
        exit()

    # load model
    embedding_matrix = np.load(embedding_fname)
    variant = kw['model']
    model = full_model(embedding_matrix=embedding_matrix, variant=variant)
    model.load_weights(weights_fname)

    # load data
    cards = pd.read_csv('processed_sets.csv', sep='\t')

    _, _, m_test, _ = get_train_test_split(cards, FULL_INPUTS)
    _, _, x_test, y_test = get_train_test_split(cards, ['text'])

    # tokenize test set
    tokenizer = Tokenizer()
    with open(tokenizer_fname, 'rb') as handle:
        tokenizer = pickle.load(handle)

    train_split = 0.8
    valid_split = 0.1
    test_split = 1 - train_split - valid_split
    frac = int((valid_split / (valid_split + test_split)) * len(x_test))

    m_test = m_test[frac:]
    x_test = x_test[frac:]
    y_test = y_test[frac:]

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=MAXLEN)

    visualize(m_test, x_test, y_test, model, variant=variant)
