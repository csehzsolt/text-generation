import os.path
import sys

import numpy
from keras.models import load_model

from src.help import get_reverse_mapping, get_data, get_n_vocab

def generate():
    # load the network weights
    checkpoint_path = 'checkpoints'
    filename = 'weights-improvement-01-2.9975.hdf5'

    model = load_model(os.path.join(checkpoint_path, filename))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    int_to_char = get_reverse_mapping()

    dataX = get_data()[0]
    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print('seed:')
    print(''.join([int_to_char[value] for value in pattern]))

    # generate characters
    for i in range (1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(get_n_vocab())
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]