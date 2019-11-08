from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.metrics import categorical_accuracy

from rnn_constants import MAXLEN

def create_model(maxlen=MAXLEN):
    model = Sequential()
    model.add(Embedding(input_dim=maxlen, output_dim=64))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='softmax'))

    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=[categorical_accuracy]
    )
    return model
