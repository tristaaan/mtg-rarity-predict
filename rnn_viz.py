import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn.metrics import confusion_matrix
from utils import get_train_test_split, plot_confusion_matrix
from rnn_model import create_model

if __name__ == '__main__':
    # load model
    maxlen = 1000
    model = create_model(maxlen)
    model.load_weights('tmp/weights-rnn.hdf5')

    # load data
    cards = pd.read_csv('processed_sets.csv', sep='\t')

    _, _, x_test, y_test = get_train_test_split(cards, ['text'])

    # tokenize test set
    tokenizer = Tokenizer(num_words=maxlen)
    tokenizer.fit_on_texts(x_test)

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=maxlen)

    # viz accuracy
    print('predicting test set...')
    y_pred = model.predict_classes(x_test)
    conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred)
    plot_confustion_matrix(conf_mat, ['common', 'uncommon', 'rare', 'mythic'], 'RNN')