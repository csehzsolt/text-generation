import os.path
from functools import reduce

res_path = 'res'
filename = 'piszkos_fred.txt'

class Text:

    def __init__(self, seq_length):
        self.raw_text = open(os.path.join(res_path, filename), encoding='utf8').read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_to_int = dict((c, i) for i, c in enumerate(self.chars))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.chars))
        self.dataX, self.dataY = self.__prepare_data(seq_length)

        self.__log()

    def __prepare_data(self, seq_length):
        '''Generate input-output pairs. All characters are converted to integers.'''
        dataX = []
        dataY = []
        for i in range(0, len(self.raw_text) - seq_length):
            seq_in = self.raw_text[i : i + seq_length]
            seq_out = self.raw_text[i + seq_length]
            dataX.append([self.char_to_int[char] for char in seq_in])
            dataY.append(self.char_to_int[seq_out])

        return dataX, dataY

    def __log(self):
        '''Log metadata about the loaded text.'''
        print('total characters: ', len(self.raw_text))
        print('total vocabulary: ', len(self.chars))
        print('total patterns: ', len(self.dataX))

        vocabulary = reduce(lambda x, y: x+y, self.chars, '')
        print('vocabulary: ', repr(vocabulary)[1:-1])
