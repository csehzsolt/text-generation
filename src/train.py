import os.path

import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from src.text import Text

def train():
    text = Text()

    # summerize the dataset
    n_chars = len(text.raw_text)
    n_vocab = len(text.chars)
    print('total characters: ', n_chars)
    print('total vocab: ', n_vocab)

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX, dataY = text.get_prepared_data(seq_length)
    n_patterns = len(dataX)
    print('total patterns: ', n_patterns)

    # reshape X to be [samples, timesteps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # define the checkpoint
    checkpoint_path = os.path.join('checkpoints', 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]

    model.fit(X, y, epochs=20, batch_size=128, callbacks=callback_list)