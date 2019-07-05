import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def get_train_test_split(df):
    '''
    Split data into training and test sets
    '''
    rs = 123
    rows = df.shape[0]
    # Shuffle data frame
    df = df.sample(frac=1, random_state=rs)

    # Select same number of samples per class for train set, remaining go to test set
    num_of_train_inputs = int(rows * 0.8)
    type_cmc = ['type', 'cmc', 'legendary']
    full_inputs = ['type', 'C', 'R', 'U', 'B', 'G', 'W', 'X', 'S',  \
                   'B/G', 'B/R', 'G/U', 'G/W', 'R/G', 'R/W', 'U/B', \
                   'U/R', 'W/B', 'W/U', 'legendary']
    inputs = full_inputs
    train_df, test_df = (pd.DataFrame(columns=inputs), pd.DataFrame(columns=inputs))
    rarities = np.unique(df.loc[:,['rarity']].values)
    for rarity in rarities:
        train_df = train_df.append(df[:num_of_train_inputs].loc[df['rarity'] == rarity, inputs+['rarity']], ignore_index=True)
        test_df  =  test_df.append(df[num_of_train_inputs:].loc[df['rarity'] == rarity, inputs+['rarity']], ignore_index=True)

    # Shuffle data frames (because they were appended in an orderly per label fashion)
    train_df = train_df.sample(frac=1, random_state=rs)
    test_df  = test_df.sample(frac=1, random_state=rs)

    types = np.unique(df.loc[:,['type']].values).tolist()

    # Split train and test datasets into labels/features
    train_values = train_df[inputs].values
    for r in train_values:
        r[0] = types.index(r[0])
    train_labels = train_df['rarity'].values.ravel()

    test_values = test_df[inputs].values
    for r in test_values:
        r[0] = types.index(r[0])
    test_labels = test_df['rarity'].values.ravel()

    return (train_values, train_labels, test_values, test_labels)


def plot_confusion_matrix(cm, labels):
    '''
    Plot confusion matrix of the specified accuracies and labels
    '''
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(6,6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    # plt.colorbar()

    # Draw ticks
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > 0.55 else "black")

    plt.tight_layout()
    plt.ylabel('True rarity')
    plt.xlabel('Predicted rarity\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    fig.subplots_adjust(bottom=0.2)
    fig.savefig('naive_confusion_matrix.png')

if __name__ == '__main__':
    cards = pd.read_csv('processed_sets.csv', sep='\t')
    train_values, train_labels, test_values, test_labels = get_train_test_split(cards)
    mlp = MLPClassifier(solver='adam', activation='tanh',
                        hidden_layer_sizes=(50), max_iter=5000,
                        random_state=123)
    mlp.fit(train_values, train_labels)

    predicted_labels = mlp.predict(test_values)
    mlp_acc = accuracy_score(test_labels, predicted_labels)
    mlp_cm = confusion_matrix(test_labels, predicted_labels)

    labels = np.unique(cards.loc[:,['rarity']].values).tolist()
    plot_confusion_matrix(mlp_cm, labels)