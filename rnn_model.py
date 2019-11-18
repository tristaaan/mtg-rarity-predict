import tensorflow as tf
tf.set_random_seed(123)

from keras import backend
from keras.initializers import Constant
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, LSTM, CuDNNLSTM, \
    Conv1D, MaxPooling1D, AveragePooling1D, concatenate
from keras.metrics import categorical_accuracy

from rnn_constants import MAXLEN, MAXFEAT, FULL_INPUTS

def simple_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAXFEAT, output_dim=64))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=[categorical_accuracy]
    )
    return model

def full_model(embedding_matrix=None, model_type='lstm'):
    # mana, type, description pipeline
    shape = len(FULL_INPUTS)
    mana_input = Input(shape=(shape,), name='costs')
    x = Dense(64, activation='relu', name='costs_fc_1')(mana_input)
    x = Dense(64, activation='relu', name='costs_fc_2')(x)
    x = Dense(64, activation='relu', name='costs_fc_3')(x)
    mana_pipeline = Dense(64, activation='relu', name='costs_fc_4')(x)

    # description pipeline
    desc_input = Input(shape=(MAXLEN,), name='description')
    # load embedding matrix if exists
    if embedding_matrix is not None:
        embed = Embedding(MAXFEAT, embedding_matrix.shape[1],
            embeddings_initializer=Constant(embedding_matrix),
            input_length=MAXLEN,
            trainable=False
        )(desc_input)
    # otherwise use basic RNN embedding
    else:
        embed = Embedding(input_dim=MAXFEAT, output_dim=64)(desc_input)

    if model_type == 'conv':
        x = Conv1D(128, kernel_size=3, activation='relu')(embed)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(3)(x)
    else:
        x = Conv1D(64, kernel_size=3, activation='relu')(embed)
        x = AveragePooling1D(3)(x)
        units = 128
        if len(backend.tensorflow_backend._get_available_gpus()):
            x = CuDNNLSTM(units)(x)
        else:
            x = LSTM(units)(x)
    else:
        print('unrecognized model type "%s"' % model_type)

    text_pipeline = Dropout(0.5)(x)
    if model_type == 'conv':
        text_pipeline = Flatten()(x)

    # concatenate and add FC layers
    cat = concatenate([mana_pipeline, text_pipeline])
    x = Dense(256, activation='relu', name='fc_1')(cat)
    x = Dense(256, activation='relu', name='fc_2')(x)
    x = Dropout(0.5, seed=123)(x)
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
