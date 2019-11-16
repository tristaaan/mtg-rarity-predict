import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from utils import get_train_test_split, plot_confusion_matrix, normalize_costs


def rf_classification(train_values, train_labels):
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=123,
        class_weight={'common':0.1, 'uncommon':1, 'rare':1, 'mythic':0.1}
    )
    rf.fit(train_values, train_labels)
    return rf


def mlp_classification(train_values, train_labels):
    mlp = MLPClassifier(solver='adam', activation='relu',
                        hidden_layer_sizes=(64,64,64,64), max_iter=10000,
                        random_state=123)
    mlp.fit(train_values, train_labels)
    return mlp


def svm_classification(train_values, train_labels):
    clf = LinearSVC(max_iter=1000)
    svm = CalibratedClassifierCV(clf)
    svm.fit(train_values, train_labels)
    return svm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive card predicion')
    parser.add_argument('-classifier', '-c', help='choose classifier',
                        default='mlp')
    args = parser.parse_args()
    kw = vars(args)
    method = kw['classifier']

    #type_cmc = ['type', 'cmc', 'legendary']
    full_inputs = ['type', 'C', 'R', 'U', 'B', 'G', 'W', 'X',  \
                   'B/G', 'B/R', 'G/U', 'G/W', 'R/G', 'R/W', 'U/B', \
                   'U/R', 'W/B', 'W/U', 'legendary', 'text']

    # load dataset
    cards = pd.read_csv('processed_sets.csv', sep='\t')
    train_values, train_labels, test_values, test_labels = \
        get_train_test_split(cards, full_inputs)

    # choose and train model
    if method == 'mlp':
        print('classifying with MLP...')
        model = mlp_classification(train_values, train_labels)
    elif method == 'rf':
        print('classifying with random forest...')
        model = rf_classification(train_values, train_labels)
    elif method == 'svm':
        print('classifying with SVM...')
        model = svm_classification(train_values, train_labels)
    else:
        print('Unrecognized classifier: "%s"' % method)
        exit()

    predicted_labels = model.predict(test_values)
    sorted_labels = ['common', 'uncommon', 'rare', 'mythic']
    acc = accuracy_score(test_labels, predicted_labels)
    cm = confusion_matrix(test_labels, predicted_labels, sorted_labels)

    labels = np.unique(cards.loc[:,['rarity']].values).tolist()
    plot_confusion_matrix(cm, sorted_labels, method)
    print('accuracy with %s: %0.2f' % (method.upper(), acc))
    # krasis = np.array([[2, 4, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]])
    # print(model.predict_proba(krasis))
