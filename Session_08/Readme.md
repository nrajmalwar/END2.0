Group Members: Nishad, Dinesh, Soma, Bharath

# Modern way of building NLP Data Pipeline using torchtext
The objective is to refactor the following models using the modern way of building data pipeline using torchtext instead of torchtext.legacy.
* Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
* Neural Machine Translation by Jointly Learning to Align and Translate

Reference link to the above models - https://github.com/bentrevett/pytorch-seq2seq

In the following sections, we will see a contrast between the old way of building data pipeline using torchtext.legacy that will be **commented** and the modern way using only torchtext.

## Import torchtext classes and functions
```python
# from torchtext.legacy.datasets import Multi30k
# from torchtext.legacy.data import Field, BucketIterator

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List
```

## Define Source and Target language
```python
# spacy_de = spacy.load('de_core_news_sm')
# spacy_en = spacy.load('en_core_web_sm')

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
```

## Tokenization
* We create a dictionary of token transforms to store the tokenizer function seperately for each language
```python
# SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
# TRG = Field(tokenize = tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

token_transform = {}

# Create source and target language tokenizer. Make sure to install the dependencies.
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en')
```

## Building vocab
* yield_tokens() is a lazy function that is used to build the vocabulary
* We store the special symbols and their indices that will be automatically used in the vocab
* vocab_transform is a dictionary used to store the vocab of each language as key and value pairs
* vocab_tranform is built using build_vocab_from_iterator()
* We use the <unk> token when a token is not found in the vocabulary
```python
# SRC.build_vocab(train_data, min_freq = 2)
# TRG.build_vocab(train_data, min_freq = 2)

vocab_transform = {}

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])
	
# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  # Training data Iterator 
  train_iterator = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  
  # Create torchtext's Vocab object 
  vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iterator, ln),
                                                    min_freq=2,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found. 
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary. 
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)
```

## Model Architecture
* The code for model architecture remains the same as written by the original author. The links to the code are-
> Model 1 - [1. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb)
>
> Model 2 - [2. Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)

## Model Object Input Dimensions
```python
# INPUT_DIM = len(SRC.vocab)
# OUTPUT_DIM = len(TRG.vocab)

INPUT_DIM = len(vocab_transform[SRC_LANGUAGE])
OUTPUT_DIM = len(vocab_transform[TGT_LANGUAGE])
```

## Loss Function
*  Pad token idx is already predefined in the modern way
```python
# TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
# criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

loss_fn = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
```

## Collate Function
* Our data iterator yields a pair of raw strings. 
* We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network defined previously. 
* Below we define our collate function that convert batch of raw strings into batch tensors thatcan be fed directly into our model. 
```python
from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
```

## Dataloader vs Iterators
* We use dataloader objects instead of iterator objects in model training and evaluation. The dataloader objects will be defined later directly in the train and evaluation functions.
```python
# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data), 
#     batch_size = BATCH_SIZE, 
#     device = device)

train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

test_iter = Multi30k(split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
```

## Training Logs
* Model 1 -
```
Epoch: 01 | Time: 1m 21s
	Train Loss: 5.018 | Train PPL: 151.165
	 Val. Loss: 4.427 |  Val. PPL:  83.704
Epoch: 02 | Time: 1m 21s
	Train Loss: 4.329 | Train PPL:  75.900
	 Val. Loss: 4.176 |  Val. PPL:  65.118
Epoch: 03 | Time: 1m 20s
	Train Loss: 4.064 | Train PPL:  58.234
	 Val. Loss: 3.927 |  Val. PPL:  50.772
Epoch: 04 | Time: 1m 21s
	Train Loss: 3.807 | Train PPL:  45.001
	 Val. Loss: 3.803 |  Val. PPL:  44.856
Epoch: 05 | Time: 1m 20s
	Train Loss: 3.529 | Train PPL:  34.106
	 Val. Loss: 3.493 |  Val. PPL:  32.895
Epoch: 06 | Time: 1m 20s
	Train Loss: 3.228 | Train PPL:  25.229
	 Val. Loss: 3.343 |  Val. PPL:  28.292
Epoch: 07 | Time: 1m 20s
	Train Loss: 2.999 | Train PPL:  20.067
	 Val. Loss: 3.173 |  Val. PPL:  23.876
Epoch: 08 | Time: 1m 20s
	Train Loss: 2.748 | Train PPL:  15.614
	 Val. Loss: 3.093 |  Val. PPL:  22.046
Epoch: 09 | Time: 1m 20s
	Train Loss: 2.539 | Train PPL:  12.662
	 Val. Loss: 2.939 |  Val. PPL:  18.903
Epoch: 10 | Time: 1m 20s
	Train Loss: 2.344 | Train PPL:  10.426
	 Val. Loss: 2.992 |  Val. PPL:  19.916
```
* Model 2 - 
```
Epoch: 01 | Time: 2m 54s
	Train Loss: 5.006 | Train PPL: 149.316
	 Val. Loss: 4.384 |  Val. PPL:  80.120
Epoch: 02 | Time: 2m 52s
	Train Loss: 4.142 | Train PPL:  62.922
	 Val. Loss: 3.775 |  Val. PPL:  43.589
Epoch: 03 | Time: 2m 53s
	Train Loss: 3.480 | Train PPL:  32.448
	 Val. Loss: 3.154 |  Val. PPL:  23.422
Epoch: 04 | Time: 2m 53s
	Train Loss: 2.925 | Train PPL:  18.626
	 Val. Loss: 2.858 |  Val. PPL:  17.435
Epoch: 05 | Time: 2m 53s
	Train Loss: 2.534 | Train PPL:  12.610
	 Val. Loss: 2.648 |  Val. PPL:  14.119
Epoch: 06 | Time: 2m 53s
	Train Loss: 2.222 | Train PPL:   9.222
	 Val. Loss: 2.587 |  Val. PPL:  13.291
Epoch: 07 | Time: 2m 52s
	Train Loss: 1.997 | Train PPL:   7.369
	 Val. Loss: 2.480 |  Val. PPL:  11.944
Epoch: 08 | Time: 2m 52s
	Train Loss: 1.785 | Train PPL:   5.962
	 Val. Loss: 2.554 |  Val. PPL:  12.857
Epoch: 09 | Time: 2m 52s
	Train Loss: 1.636 | Train PPL:   5.136
	 Val. Loss: 2.459 |  Val. PPL:  11.694
Epoch: 10 | Time: 2m 52s
	Train Loss: 1.508 | Train PPL:   4.516
	 Val. Loss: 2.557 |  Val. PPL:  12.891
```
