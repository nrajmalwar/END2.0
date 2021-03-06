# END2.0
Assignment submissions from the END2.0 program. Deep NLP with RNNs/LSTMs, Attention Mechanism and Transformers from scratch.

## [Session 1: Background & Very Basics](Session_01)

* Rewrite Google Colab file with the specified changes
* Answer questions on the basics of neural networks

## [Session 2: BackProp, Embeddings and Language Models ](Session_02)

* Train a neural network on excel sheet with all the backpropagation calculations involved

## [Session 3: Pytorch](Session_03)

Write a neural network that takes two inputs
* an image from MNIST dataset
* a random number between 0 and 9

and gives the output
* the "number" that was represented by the MNIST image
* the "sum" of this number with the random number that was generated and sent as the input to the network

## [Session 4: RNNs and LSTMs](Session_04)

* Train a simple classification network on the IMDb dataset using LSTM

## [Session 6: GRUs, Seq2Seq and Introduction to Attention Mechanism](Session_06)

* Train a classification model on the tweets dataset using the encoder-decoder architecture
* Visualize the encoding and decoding vectors at each time step

## [Session 7: Second Hands-on](Session_07)
* [Part1](Session_07/Part1_SST_Classification) - Train a classification model on the Stanford Sentiment Tree (SST) Bank Dataset
* [Part2](Session_07/Part2_Seq2Seq_Datasets) - Train a Seq2Seq architecture model on the following datasets-
    1. Quora dataset
    2. Question Answer dataset

## [Session 8: torchtext & Advanced Concepts](Session_08)
* Refactor the following Seq2Seq models using the modern way of building data pipelines using torchtext (instead of torchtext.legacy)-
    1. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](Session_08/2_Learning_Phrase_Representations_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation.ipynb)
    2. [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/nrajmalwar/END2.0/blob/main/Session_08/3_Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate.ipynb)

## [Session 9: Learning Rates and Evaluation Metrics - Part 1](Session_09)
* Implement the following evaluation metrics-
1. [For classification task (tweets dataset)](Session_09/Precision_Recall_F1.ipynb) -
    i. Precision, Recall and F1 Score
2. [For language translation Seq2Seq model (Multi30k dataset)](Session_09/Bleu_Perplexity_Bert_Score.ipynb) -
    ii. BLEU score
    iii. Perplexity
    iv. BERTScore
