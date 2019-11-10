from tensorflow import set_random_seed
set_random_seed(123)

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Input, LSTM, concatenate
from keras.metrics import categorical_accuracy

from rnn_constants import MAXLEN, FULL_INPUTS

def create_model(maxlen=MAXLEN):
    # mana, type, description pipeline
    shape = len(FULL_INPUTS)
    mana_input = Input(shape=(shape,), name='costs')
    mana_fc_1 = Dense(64, activation='relu', name='costs_dense')
    mana_pipeline = mana_fc_1(mana_input)

    # description pipeline
    desc_input = Input(shape=(maxlen,), name='description')
    embed = Embedding(input_dim=maxlen, output_dim=64)
    lstm = LSTM(128, dropout=0.2, recurrent_dropout=0.2)
    text_pipeline = lstm(embed(desc_input))

    # concatenate and add FC layers
    cat = concatenate([mana_pipeline, text_pipeline])
    fc_1 = Dense(512, activation='relu', name='fc_1')(cat)
    dropout = Dropout(0.5, seed=123)(fc_1)
    fc_2 = Dense(512, activation='relu', name='fc_2')(dropout)
    output = Dense(4, activation='softmax', name='rarity_output')(fc_2)

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
    create_model()
