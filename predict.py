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