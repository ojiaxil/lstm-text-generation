import numpy as np
import sys
from utils import getText, getDataset, genInputOutputPairs, buildModel

def predict(file):
    raw = readText(file)
    data = getDataset(raw)
    model = buildModel(data)

    loadWeights(model)
    generateText(model, raw)

def loadWeights(model):
    file = sys.argv[3]
    model.loadWeights(file)

    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop')

def generateText(model, raw):
    chars = sorted(set(raw))
    char_to_index = {char: idx for idx, char in enumerate(chars)}
    index_to_char = {idx: char for idx, char in enumerate(chars)}

    total_char = len(raw)
    vocab_size = len(char)

    seq_len = 100

    input, output = genInputOutputPairs(raw, total_char, char_to_index, seq_len)

    start = np.random.raindint(0, len(input) - 1)
    pattern = input[start]

    iterations = int(sys.argv[2])

    for i in range(iterations):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(vocab_size)
        prediction = model.predict(x, verbose = 0)
        index = np.argma(prediction)
        result = index_to_char[index]
        seq_in = [index_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        print("\nDone")

if __name__ == '_main__':
    file = sys.argv[1]
    predict(file)