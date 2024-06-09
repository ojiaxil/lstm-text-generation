# LSTM Text Generation

This project enables you to train a Long Short Term Memory neural network to generate text from any TXT file with more than 100 characters.

## Train

To train the network, run `lstm.py` with your file:

```bash
python lstm.py <file-name>
```
- __file-name:__ The text file for training.
- The model will train for 200 epochs.
- Training can be stopped anytime, and the weights from the latest completed epoch will be saved for future text generation.

## Generate Text

You can generate text after training the network using `predict.py`:

```bash
python predict.py <file-name> <text-length> <weights-file>
```
- __file-name:__ The same text file used for training.
- __text-length:__ Desired length of generated text (minimum 100).
- __weights-file:__ The file containing the weights to load into the network.
