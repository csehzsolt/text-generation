import os.path

res_path = 'res'
filename = 'wonderland.txt'

class Text:

    def __init__(self):
        self.raw_text = open(os.path.join(res_path, filename)).read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_to_int = dict((c, i) for i, c in enumerate(self.chars))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.chars))

    def get_prepared_data(self, seq_length):
        dataX = []
        dataY = []
        for i in range(0, len(self.raw_text) - seq_length):
            seq_in = self.raw_text[i : i + seq_length]
            seq_out = self.raw_text[i + seq_length]
            dataX.append([self.char_to_int[char] for char in seq_in])
            dataY.append(self.char_to_int[seq_out])

        return dataX, dataY