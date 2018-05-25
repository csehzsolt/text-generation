import os.path
import sys

import numpy
from keras.models import load_model

from src.text import Text

checkpoint_path = 'checkpoints'
filename = 'weights-improvement-02-2.7863.hdf5'

def generate():
    text = Text()

    # load the network weights
    model = load_model(os.path.join(checkpoint_path, filename))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    seq_length = 100
    dataX = text.get_prepared_data(seq_length)[0]
    int_to_char = text.int_to_char
    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print('seed:')
    print(''.join([int_to_char[value] for value in pattern]))

    # generate characters
    for i in range (1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(text.chars))
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]