import argparse
import pickle
import operator
from os import path

import pandas as pd
import numpy as np
import tensorflow as tf

from jinja2 import Environment, FileSystemLoader, select_autoescape
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from rnn_model import full_model
from utils import get_all_cards
from constants import SETS, FULL_INPUTS, DEFAULT_WEIGHTS, DEFAULT_EMBEDDING, \
  DEFAULT_TOKENIZER, MAXLEN


def arg_set(s):
    arr = list(map(int, s.split(',')))
    assert len(arr) >= 0, 'there must be at least one set listed'
    for name in arr:
      assert name.upper() in SETS, 'Set name not valid: %s' % name
    return arr


def argmax(arr):
    return np.argmax(arr)


def distance(args):
    r1,r2 = args
    arr = ['c', 'u', 'r', 'm']
    return abs(arr.index(r1) - arr.index(r2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze model results')
    parser.add_argument('-model', '-m', help='model variant',
                        default='lstm')
    parser.add_argument('-weights', '-w', help='the weights to use',
                        default=DEFAULT_WEIGHTS)
    parser.add_argument('-size', '-s', help='size of word vectors',
                        type=int, default=200)
    parser.add_argument('-embedding', '-e', help='the word_embedding to use',
                        default=DEFAULT_EMBEDDING)
    parser.add_argument('-tokenizer', '-t', help='the weights to use',
                        default=DEFAULT_TOKENIZER)
    parser.add_argument('-sets', help='sets to visualize', action='store',
                        default=['GRN', 'RNA', 'WAR', 'M20', 'ELD'],
                        type=arg_set)

    args = parser.parse_args()
    kw = vars(args)

    embedding_fname = kw['embedding']
    wv_size = kw['size']
    if 'zzz' in embedding_fname:
        embedding_fname = embedding_fname.replace('zzz', str(wv_size))

    # check file existence before we load everything into memory
    if not path.isfile(embedding_fname):
        print('embedding file "%s" does not exist' % embedding_fname)
        exit()

    weights_fname = kw['weights']
    if not path.isfile(weights_fname):
        print('weights file "%s" does not exist' % weights_fname)
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

    # load data, filter by set
    cards = pd.read_csv('processed_sets.csv', sep='\t')
    cards = cards[cards['set'].isin(kw['sets'])]

    # fetch cost and descriptions
    print('Filtering for sets: %s' % kw['sets'])
    costs = get_all_cards(cards, FULL_INPUTS)
    descs = get_all_cards(cards, ['text'])


    # tokenize test set
    tokenizer = Tokenizer()
    with open(tokenizer_fname, 'rb') as handle:
        tokenizer = pickle.load(handle)

    descs = tokenizer.texts_to_sequences(descs)
    descs = pad_sequences(descs, maxlen=MAXLEN)

    predictions = model.predict([costs, descs], batch_size=48)
    env = Environment(
        loader=FileSystemLoader('./templates'),
        autoescape=select_autoescape(['html', 'xml'])
    )
    env.filters['argmax'] = argmax
    env.filters['distance'] = distance
    template = env.get_template('card-analysis.html')
    with open('rnn-analysis.html', 'w') as output:
        cards_fmt = cards.to_dict('records')
        expansions = np.unique(cards.loc[:,['set']].values).tolist()
        output.write(
            template.render(
                cards=cards_fmt,
                predictions=predictions,
                expansions=expansions
            )
        )

    print('done')
