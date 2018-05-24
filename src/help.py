import os.path

res_path = 'res'
filename = 'wonderland.txt'

raw_text = open(os.path.join(res_path, filename)).read()
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)

def get_data():
    ''' prepare the dataset of input to output pairs encoded as integers '''
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length):
        seq_in = raw_text[i : i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
        
    return dataX, dataY

def get_reverse_mapping():
    return dict((i, c) for i, c in enumerate(chars))

def get_n_vocab():
    return n_vocab