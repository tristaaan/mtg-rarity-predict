import pandas as pd
from gensim.models import Word2Vec

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    # 'first strike',
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
    # load cards
    cards = pd.read_csv('processed_sets.csv', sep='\t')
    tok_vals = cards['text'].str.split().values

    # Word2Vec
    model = Word2Vec(tok_vals, size=100, window=5, seed=1234)
    vocab = list(model.wv.vocab)
    X = model[vocab]

    # dimensionality reduction
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

    fig.savefig('card_viz_c=%d.png' % n_components)
