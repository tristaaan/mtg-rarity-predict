import argparse
import pickle

from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from sklearn.manifold import TSNE

from keras.preprocessing.text import Tokenizer
from utils import build_embedding_matrix
from rnn_constants import MAXLEN

ev_keyword_actions = [
    'activate',
    'attach',
    'cast',
    'counter',
    'create',
    'destroy',
    'discard',
    'enchant',
    'equip',
    'exchange',
    'exile',
    'fight',
    'flash',
    'play',
    'reveal',
    'sacrifice',
    'scry',
    'search',
    'shuffle',
    'tap',
    'untap',
    # additions:
    'attack',
    'block',
    'damage',
    'gain',
    'lose',
    'life'
]

ev_keywords = [
    # 'double strike', # can't annotate these
    # 'first strike', # so they're expanded
    'first',
    'double',
    'strike',
    'deathtouch',
    'defender',
    'enchant',
    'flying',
    'haste',
    'hexproof',
    'indestructible',
    'lifelink',
    'menace',
    'protection',
    'reach',
    'trample',
    'vigilance'
]

types = [
    'creature',
    'instant',
    'sorcery',
    'enchantment',
    'artifact',
    'planeswalker',
    'land',
    # additions
    'nonland',
    'noncreature'
]

lands = [
    'plain',
    'white',
    'forest',
    'green',
    'island',
    'blue',
    'swamp',
    'black',
    'mountain',
    'red',
    'colorless'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build word2vec model')
    parser.add_argument('-size', '-s', help='vector size',
                        type=int, default=100)
    args = parser.parse_args()
    kw = vars(args)
    size = kw['size']

    # load cards
    cards = pd.read_csv('processed_sets.csv', sep='\t')
    corpus = cards['text'].str.split().values
    tokenizer = Tokenizer(num_words=MAXLEN)
    tokenizer.fit_on_texts(corpus)
    # save for later?
    with open(path.join('tmp','tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Word2Vec
    print('building word2vec model with vec size %d...' % size)
    model = Word2Vec(corpus, size=size, window=5, seed=1234)
    # save model
    print('writing model...')
    build_embedding_matrix(tokenizer.word_docs.keys(), tokenizer.word_index,
                           size=size)
    vocab = list(model.wv.vocab)
    X = model[vocab]

    # dimensionality reduction
    print('visualizing model...')
    n_components = 2
    tsne = TSNE(n_components=n_components, init='pca',
                random_state=1234, method='exact')
    X_tsne = tsne.fit_transform(X)
    df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

    # visualize
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'].values, df['y'].values, marker='.')

    # annotate
    for word, pos in df.iterrows():
        if word in ev_keywords:
            ax.annotate(word, pos, color='green')
        elif word in ev_keyword_actions:
            ax.annotate(word, pos, color='darkred')
        elif word in types:
            ax.annotate(word, pos, color='tab:purple')
        elif word in lands or word[:-1] in lands:
            ax.annotate(word, pos, color='tab:orange')
        else:
            ax.annotate(word, pos, color='gray')
    fname = 'card_viz_c=%d_size=%d.png' % (n_components, size)
    print('figure saved as %s' % fname)
    fig.savefig(fname)
