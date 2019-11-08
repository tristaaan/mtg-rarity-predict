from __future__ import absolute_import, division, print_function, unicode_literals
import os

import matplotlib.pyplot as plt
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn.metrics import confusion_matrix
from utils import get_train_test_split, plot_confusion_matrix
from rnn_model import create_model

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'
tf.Session(config = config)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

def make_results_folder(name):
    try:
        os.mkdir(name)
    except FileExistsError:
        pass

if __name__ == '__main__':
    batch_size = 32
    maxlen = 1000
    cards = pd.read_csv('processed_sets.csv', sep='\t')
    x_train, y_train, x_test, y_test = get_train_test_split(cards, ['text'])

    tokenizer = Tokenizer(num_words=maxlen)
    tokenizer.fit_on_texts(x_train)
    print('Found %s unique tokens.' % len(tokenizer.word_index))

    # tokenize train and test sets
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=maxlen)

    x_test  = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=maxlen)

    make_results_folder('tmp')
    checkpointer = ModelCheckpoint(filepath=os.path.join('tmp','weights-rnn.hdf5'),
        monitor='loss',
        verbose=1,
        save_best_only=True
    )

    model = create_model(maxlen)
    model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=10,
        callbacks=[checkpointer]
    )
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('\nscore:    %.02f' % score)
    print('accuracy: %.02f\n' % acc)
