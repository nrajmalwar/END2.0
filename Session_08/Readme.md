# Modern way of building NLP Data Pipeline using torchtext
The objective is to refactor the following models using the modern way of building data pipeline using torchtext instead of torchtext.legacy.
* Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
* Neural Machine Translation by Jointly Learning to Align and Translate

Reference link to the above models - https://github.com/bentrevett/pytorch-seq2seq

In the following sections, we will see a contrast between the old way of building data pipeline using torchtext.legacy that will be **commented** and the modern way using only torchtext.

## Importing torchtext classes and functions
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
```python
# SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
# TRG = Field(tokenize = tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

token_transform = {}

# Create source and target language tokenizer. Make sure to install the dependencies.
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en')

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
```

## Building vocab
```python
# SRC.build_vocab(train_data, min_freq = 2)
# TRG.build_vocab(train_data, min_freq = 2)

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
## Dataloader vs Iterators
* We use dataloader objects instead of iterator objects in model training and evaluation. The dataloader objects will be defined later directly in the train and evaluation functions.
```python
# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data), 
#     batch_size = BATCH_SIZE, 
#     device = device)
```

