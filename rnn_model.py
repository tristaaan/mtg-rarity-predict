from tensorflow import set_random_seed
set_random_seed(123)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Input, LSTM, concatenate
from keras.metrics import categorical_accuracy

from rnn_constants import MAXLEN, FULL_INPUTS

def simple_model(maxlen=MAXLEN):
    model = Sequential()
    model.add(Embedding(input_dim=maxlen, output_dim=64))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=[categorical_accuracy]
    )
    return model

def full_model(maxlen=MAXLEN):
    # mana, type, description pipeline
    shape = len(FULL_INPUTS)
    mana_input = Input(shape=(shape,), name='costs')
    mana_fc_1 = Dense(64, activation='relu', name='costs_dense')
    mana_pipeline = mana_fc_1(mana_input)

    # description pipeline
    desc_input = Input(shape=(maxlen,), name='description')
    embed = Embedding(input_dim=maxlen, output_dim=64)
    lstm = LSTM(64, dropout=0.2, recurrent_dropout=0.2)
    text_pipeline = lstm(embed(desc_input))

    # concatenate and add FC layers
    cat = concatenate([mana_pipeline, text_pipeline])
    x = Dense(512, activation='relu', name='fc_1')(cat)
    x = Dense(512, activation='relu', name='fc_2')(x)
    x = Dropout(0.5, seed=123)(x)
    x = Dense(256, activation='relu', name='fc_3')(x)
    x = Dense(256, activation='relu', name='fc_4')(x)
    x = Dropout(0.5, seed=456)(x)
    output = Dense(4, activation='softmax', name='rarity_output')(x)

    # build model
    model = Model(inputs=[mana_input, desc_input], outputs=[output])
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=[categorical_accuracy]
    )
    return model

if __name__ == '__main__':
    full_model()
