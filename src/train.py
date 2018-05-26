import os.path

import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from src.text import Text

seq_length = 100

def train():
    text = Text(seq_length)

    dataX = text.dataX
    dataY = text.dataY

    # reshape X to be [samples, timesteps, features]
    X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
    # normalize
    X = X / float(len(text.chars))
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # define the checkpoint
    checkpoint_path = os.path.join('checkpoints', 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]

    model.fit(X, y, epochs=20, batch_size=128, callbacks=callback_list)