from __future__ import absolute_import, division, print_function, unicode_literals
import os

import matplotlib.pyplot as plt
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix

from utils import get_train_test_split, pretrained_embedding_matrix
from rnn_model import full_model
from rnn_viz import visualize
from rnn_constants import MAXLEN, FULL_INPUTS
from utils import normalize_costs

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'
tf.Session(config = config)


def plot_graphs(history):
    fig = plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'], '')
    plt.xlabel("Epochs")
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'])
    fname = 'rnn-training-curve.png'
    fig.savefig(fname)
    print('Training curves written as: "%s"' % fname)


def make_results_folder(name):
    try:
        os.mkdir(name)
    except FileExistsError:
        pass


if __name__ == '__main__':
    batch_size = 32

    # load data
    cards = pd.read_csv('processed_sets.csv', sep='\t')

    # split data
    train_split = 0.7
    manas_train, _, manas_test, _ = get_train_test_split(cards, FULL_INPUTS,
            train_split)
    x_train, y_train, x_test, y_test = get_train_test_split(cards, ['text'],
            train_split)

    manas_train = normalize_costs(manas_train)
    manas_test = normalize_costs(manas_test)

    # split the test set into validation and test sets
    frac = int(((1-train_split) / 2) * len(x_test))
    x_valid = x_test[:frac]
    y_valid = y_test[:frac]

    manas_valid = manas_test[:frac]
    manas_test = manas_test[frac:]

    x_test = x_test[frac:]
    y_test = y_test[frac:]

    # tokenize descriptions
    corpus = cards['text'].str.split().values
    tokenizer = Tokenizer(num_words=MAXLEN)
    tokenizer.fit_on_texts(corpus)
    print('Found %s unique tokens.' % len(tokenizer.word_index))

    embedding_mat = pretrained_embedding_matrix(
        corpus,
        tokenizer.word_index
    )

    # tokenize train, validation, and test sets
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=MAXLEN)

    x_valid = tokenizer.texts_to_sequences(x_valid)
    x_valid = pad_sequences(x_valid, maxlen=MAXLEN)

    x_test  = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=MAXLEN)

    # save model as we go
    make_results_folder('tmp')
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('tmp','weights-rnn.hdf5'),
        monitor='loss',
        verbose=1,
        save_best_only=True
    )

    # train model
    model = full_model(embedding_matrix=embedding_mat)
    hist = model.fit([manas_train, x_train], y_train,
        validation_data=([manas_valid, x_valid], y_valid),
        batch_size=batch_size,
        epochs=10,
        callbacks=[checkpointer]
    )

    # evaluate and visualize
    score, acc = model.evaluate([manas_test, x_test], y_test,
                                batch_size=batch_size)
    print('\nscore:    %.02f' % score)
    print('accuracy: %.02f\n' % acc)
    plot_graphs(hist)
    visualize(manas_test, x_test, y_test, model)
