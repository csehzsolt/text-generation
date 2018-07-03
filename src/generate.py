import os.path
import sys

import numpy
from keras.models import load_model

from src.text import Text

checkpoint_path = 'checkpoints'
filename = 'weights-improvement-04-2.3871.hdf5'

seq_length = 100
number_of_predicted_characters = 500

def generate():
    text = Text(seq_length)

    # load the network weights
    model = load_model(os.path.join(checkpoint_path, filename))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    dataX = text.dataX
    int_to_char = text.int_to_char
    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print('seed:')
    print(''.join([int_to_char[value] for value in pattern]))

    # generate characters
    for i in range (number_of_predicted_characters):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(text.chars))
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]