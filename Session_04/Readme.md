# LSTM Network for Classification on IMDB Dataset
The objective is to replace RNN layer with an LSTM layer in the network

## Input Data Pre-processing

*  Text is tokenized using the spacy library and the 'en_core_web_sm' tokenizer language
*  Labels uses the data type torch.float
*  Split the dataset into train and test for the IMDB Dataset
*  Further split the train data into train and valid set
*  Maximum vocabulary is set to 25,000 words
*  Create iterator objects for the train, valid and test data

## Network Design
### Layers
* Embedding layer for input
* LSTM layer
* Fully connected layer

### Forward Pass
* Pack Padded Sequence to create input to the model of uniform length for each batch. The length of sequence is selected using the maximum length of sequence in a batch

### Model Parameters
```
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
```
> Number of parameters in the model = 2,867,049 trainable parameters

* Learning rate is Adam 1e-3
* Loss is BCEWithLogitsLoss()

## Model Training
* Model is trained for 10 epochs

```
Epoch: 01 | Epoch Time: 0m 26s
	Train Loss: 0.683 | Train Acc: 55.94%
	 Val. Loss: 0.668 |  Val. Acc: 58.88%
Epoch: 02 | Epoch Time: 0m 25s
	Train Loss: 0.600 | Train Acc: 67.59%
	 Val. Loss: 0.546 |  Val. Acc: 73.60%
Epoch: 03 | Epoch Time: 0m 25s
	Train Loss: 0.485 | Train Acc: 77.50%
	 Val. Loss: 0.505 |  Val. Acc: 75.64%
Epoch: 04 | Epoch Time: 0m 25s
	Train Loss: 0.378 | Train Acc: 83.94%
	 Val. Loss: 0.619 |  Val. Acc: 71.74%
Epoch: 05 | Epoch Time: 0m 25s
	Train Loss: 0.305 | Train Acc: 87.96%
	 Val. Loss: 0.440 |  Val. Acc: 82.72%
Epoch: 06 | Epoch Time: 0m 25s
	Train Loss: 0.247 | Train Acc: 90.44%
	 Val. Loss: 0.415 |  Val. Acc: 83.17%
Epoch: 07 | Epoch Time: 0m 25s
	Train Loss: 0.221 | Train Acc: 92.07%
	 Val. Loss: 0.423 |  Val. Acc: 84.63%
Epoch: 08 | Epoch Time: 0m 25s
	Train Loss: 0.153 | Train Acc: 94.70%
	 Val. Loss: 0.457 |  Val. Acc: 84.04%
Epoch: 09 | Epoch Time: 0m 25s
	Train Loss: 0.140 | Train Acc: 95.24%
	 Val. Loss: 0.454 |  Val. Acc: 83.07%
Epoch: 10 | Epoch Time: 0m 25s
	Train Loss: 0.123 | Train Acc: 95.74%
	 Val. Loss: 0.489 |  Val. Acc: 84.74%
```

## Model Evaluation
```
Test Loss: 0.408 | Test Acc: 83.56%
```

## Observations
The model improves significantly upon using LSTM layers whereas the RNN had the accuracy stuck at 50%, which is a random guess accuracy for a binary classification problem.
