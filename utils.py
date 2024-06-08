import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.utils import np_utils

def getText(file):
    raw = open(file, encoding = "utf8").read()
    return raw

# create a dataset of sequences and corresponding outputs

def prep_dataset(raw):
    unique_char = sorted(set(raw))
    char_to_index = {char: idx for idx, char in enumerate(unique_char)}

    # calculate total num of characters and unique vocab size

    total_char = len(raw)
    vocab_size = len(unique_char)

    print(f"Total Characters: {total_char}")
    print(f"Total Vocabulary: {vocab_size}")

    # define sequence length input-output pairs

    seq_len = 100

    inputs, outputs = gen_input_output_pairs(raw, total_char, char_to_index, seq_len)

    num_patterns = len(inputs)
    print(f"Total Patterns: {num_patterns}")

    input_data = np.reshape(inputs, (num_patterns, seq_len, 1))
    input_data = input_data / float(vocab_size)
    output_data = np_utils.to_categorical(outputs)

    return {'vocab_size': vocab_size, 'input_data': input_data, 'output_data': output_data}

# genergate input-output pairs for sequence data

def gen_input_output_pairs(raw, total_char, char_to_index, seq_len):
    inputs = []
    outputs = []

    for i in range(total_char - seq_len):
        seq_in = raw[i:i + seq_len]
        seq_out = raw[i + seq_len]
        inputs.append([char_to_index[char] for char in seq_in])
        outputs.append(char_to_index[seq_out])

    return inputs, outputs
