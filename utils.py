import itertools
import os

import numpy as np
np.random.seed(123)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from rnn_constants import MAXFEAT, RARITIES

def get_train_test_split(df, inputs, train_split=0.8):
    '''
    Split data into training and test sets
    '''
    rs = 123
    rows = df.shape[0]
    # Shuffle data frame
    df = df.sample(frac=1, random_state=rs)

    assert train_split < 1, 'Train split must be less than 1'
    test_split = 1 - train_split

    # Select same number of samples per class for train set, remaining go to test set
    # num_of_train_inputs = int(rows * 0.8)
    min_rarity_count = get_min_rarity_count(df)
    train_count = int(min_rarity_count * train_split)
    test_count  = int(min_rarity_count * test_split)
    print('train: %d, test: %d' % (train_count*4, test_count*4))

    train_df, test_df = (pd.DataFrame(columns=inputs), pd.DataFrame(columns=inputs))
    for rarity in RARITIES:
        train_df = train_df.append(df.loc[df['rarity'] == rarity, inputs+['rarity']][:train_count], ignore_index=True)
        test_df  =  test_df.append(df.loc[df['rarity'] == rarity, inputs+['rarity']][train_count:train_count+test_count], ignore_index=True)

    # Shuffle data frames (because they were appended in an orderly per label fashion)
    train_df = train_df.sample(frac=1, random_state=rs)
    test_df  = test_df.sample(frac=1, random_state=rs)

    # encode spell type as integer.
    spell_types = np.unique(df.loc[:,['type']].values).tolist()

    # Split train and test datasets into features/labels
    if 'text' in inputs and len(inputs) == 1:
        train_values, train_labels = text_split(train_df, spell_types)
        test_values, test_labels   = text_split(test_df, spell_types)
    else:
        train_values, train_labels = split(train_df, inputs, spell_types)
        test_values, test_labels   = split(test_df, inputs, spell_types)

    return (train_values, train_labels, test_values, test_labels)


def text_split(df, spell_types):
    # print(df[columns].values.tolist()[0:5])
    values = df['text'].str.split().values
    labels = df['rarity'].values.ravel()
    labels = np.array(list(map(label_array, labels)))
    return (values, labels)


def label_array(rarity):
    ret = [0, 0, 0, 0]
    ind = RARITIES.index(rarity)
    ret[ind] = 1
    return ret


def split(df, columns, spell_types):
    if 'text' in columns:
        df['text'] = df['text'].apply(lambda x: len(x.split()))
    values = df[columns].values
    # convert spell type into numerical value
    for r in values:
        # len(spell_types) // 2 to normalize spell type
        r[0] = spell_types.index(r[0]) - len(spell_types)//2
    labels = df['rarity'].values.ravel()
    return (values, labels)


def normalize_costs(df):
    return (df - df.mean()) / (df.max() - df.min())


def pretrained_embedding_matrix(texts, word_index):
    # load cache if exists
    glove_dir = '.'
    cache_dir = 'tmp'
    fname = 'embedding_matrix.npy'
    cache_path = os.path.join(cache_dir, fname)
    if os.path.isfile(cache_path):
        print('Using cached embedding matrix...')
        return np.load(cache_path)

    print('Building embedding matrix...')
    embeddings_index = {}
    size = 200
    with open(os.path.join(glove_dir, 'glove.6B.%dd.txt' % size), encoding='UTF8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    embedding_matrix = np.random.normal(size=(MAXFEAT, size))

    non = 0
    # num = 0
    for word, index in word_index.items():
        # if word in '1234567890':
        #     num += 1
        #     continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
        else:
            non += 1
    print(non, 'words did not have embed mappings')
    # print(num, 'words were numerics')

    np.save(os.path.join(cache_dir, fname), embedding_matrix)
    return embedding_matrix


def get_min_rarity_count(cards):
    rarities = np.unique(cards.loc[:,['rarity']].values)
    counts = cards['rarity'].value_counts(dropna=False)
    print(counts)
    return min(counts)


def make_folder(name):
    try:
        os.mkdir(name)
    except FileExistsError:
        pass


def plot_confusion_matrix(cm, labels, method):
    '''
    Plot confusion matrix of the specified accuracies and labels
    '''
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    # cmap = plt.get_cmap('Blues')
    fig = plt.figure(figsize=(6,6))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize
    plt.imshow(cm, interpolation='nearest', cmap='Blues',
               norm=colors.Normalize(vmin=0.0, vmax=1.0))
    plt.title('Confusion matrix - %s' % method.upper())
    plt.colorbar()

    # Draw ticks
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Label values
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > 0.55 else "black")

    fname = 'confusion_matrix-%s.png' % method
    plt.tight_layout()
    plt.ylabel('True rarity')
    plt.xlabel('Predicted rarity\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    fig.subplots_adjust(bottom=0.1, left=0.2)
    fig.savefig(fname)
    print('Confusion matrix written as: "%s"' % fname)
