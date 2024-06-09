import sys
from keras.callbacks import ModelCheckpoint
from utils import getDataset, buildModel, getText

def train(file):
    raw = getText(file)
    data = getDataset(raw)
    model = buildModel(data)
    fitModel(model, data)

def fitModel(model, data):
    path = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(path, monitor = 'loss', verbose = 0, best = True, mode = 'min')
    callbacks_list = [checkpoint]

    model.fit(data['input'], data['output'], nb_epoch = 200, batch_size = 64, callbacks = callbacks_list)

if __name__ == "__main__":
    file = sys.argv[1]
    train(file)