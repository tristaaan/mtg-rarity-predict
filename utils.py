import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def get_train_test_split(df, inputs):
    '''
    Split data into training and test sets
    '''
    rs = 123
    rows = df.shape[0]
    # Shuffle data frame
    df = df.sample(frac=1, random_state=rs)

    # Select same number of samples per class for train set, remaining go to test set
    # num_of_train_inputs = int(rows * 0.8)
    num_of_train_inputs = int(get_min_rarity_count(df) * 0.8)
    print(num_of_train_inputs)

    train_df, test_df = (pd.DataFrame(columns=inputs), pd.DataFrame(columns=inputs))
    rarities = np.unique(df.loc[:,['rarity']].values)
    print(rarities)
    for rarity in rarities:
        train_df = train_df.append(df[:num_of_train_inputs].loc[df['rarity'] == rarity, inputs+['rarity']], ignore_index=True)
        test_df  =  test_df.append(df[num_of_train_inputs:].loc[df['rarity'] == rarity, inputs+['rarity']], ignore_index=True)

    # Shuffle data frames (because they were appended in an orderly per label fashion)
    train_df = train_df.sample(frac=1, random_state=rs)
    test_df  = test_df.sample(frac=1, random_state=rs)

    # encode spell type as integer.
    spell_types = np.unique(df.loc[:,['type']].values).tolist()

    # Split train and test datasets into features/labels
    if 'text' in inputs:
        train_values, train_labels = text_split(train_df, inputs, spell_types)
        test_values, test_labels = text_split(test_df, inputs, spell_types)  
    else:
        train_values, train_labels = split(train_df, inputs, spell_types)
        test_values, test_labels = split(test_df, inputs, spell_types)

    return (train_values, train_labels, test_values, test_labels)


def text_split(df, columns, spell_types):
    # print(df[columns].values.tolist()[0:5])
    values = list(map(lambda x: x[0].split(), df[columns].values.tolist()))
    labels = df['rarity'].values.ravel()
    # print(labels[0:5])
    labels = np.array(list(map(label_array, labels)))
    return (values, labels)


def label_array(rarity):
    ret = [0, 0, 0, 0]
    rarities = ['common', 'mythic', 'rare', 'uncommon']
    ind = rarities.index(rarity)
    ret[ind] = 1
    return ret


def split(df, columns, spell_types):
    values = df[columns].values
    # convert spell type into numerical value
    for r in values:
        # len(spell_types) // 2 to normalize spell type
        r[0] = spell_types.index(r[0]) - len(spell_types)//2
    labels = df['rarity'].values.ravel()
    return (values, labels)


def get_min_rarity_count(cards):
    rarities = np.unique(cards.loc[:,['rarity']].values)
    counts = cards['rarity'].value_counts(dropna=False)
    print(counts)
    return min(counts)


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

    plt.tight_layout()
    plt.ylabel('True rarity')
    plt.xlabel('Predicted rarity\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    fig.subplots_adjust(bottom=0.1, left=0.2)
    fig.savefig('naive_confusion_matrix-%s.png' % method)
    