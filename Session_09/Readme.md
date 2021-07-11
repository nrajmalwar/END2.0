
## Perplexity

In general, perplexity is a measurement of how well a probability model predicts a sample. In the context of Natural Language Processing, perplexity is one way to evaluate language models.

<img width="1000" alt="image" src="./static/perplexity.png">


Less entropy (or less disordered system) is favorable over more entropy. Because predictable results are preferred over randomness. This is why people say low perplexity is good and high perplexity is bad since the perplexity is the exponentiation of the entropy (and you can safely think of the concept of perplexity as entropy).

## Training logs:
```
Epoch: 01 | Time: 0m 43s
	Train Loss: 5.035 | Train PPL: 153.672
	 Val. Loss: 4.450 |  Val. PPL:  85.589
	 BLEU Score: 2.61
Epoch: 02 | Time: 0m 42s
	Train Loss: 4.327 | Train PPL:  75.735
	 Val. Loss: 4.157 |  Val. PPL:  63.908
	 BLEU Score: 3.63
Epoch: 03 | Time: 0m 42s
	Train Loss: 4.081 | Train PPL:  59.222
	 Val. Loss: 3.988 |  Val. PPL:  53.956
	 BLEU Score: 4.07
Epoch: 04 | Time: 0m 42s
	Train Loss: 3.825 | Train PPL:  45.826
	 Val. Loss: 3.811 |  Val. PPL:  45.183
	 BLEU Score: 6.71
Epoch: 05 | Time: 0m 42s
	Train Loss: 3.534 | Train PPL:  34.278
	 Val. Loss: 3.488 |  Val. PPL:  32.734
	 BLEU Score: 9.17
Epoch: 06 | Time: 0m 42s
	Train Loss: 3.263 | Train PPL:  26.122
	 Val. Loss: 3.354 |  Val. PPL:  28.605
	 BLEU Score: 11.81
Epoch: 07 | Time: 0m 42s
	Train Loss: 3.006 | Train PPL:  20.208
	 Val. Loss: 3.188 |  Val. PPL:  24.248
	 BLEU Score: 15.04
	 BERT Score: Precision=0.892, Recall=0.890, F1 Score=0.891
Epoch: 08 | Time: 0m 43s
	Train Loss: 2.754 | Train PPL:  15.709
	 Val. Loss: 2.996 |  Val. PPL:  20.015
	 BLEU Score: 16.95
	 BERT Score: Precision=0.897, Recall=0.894, F1 Score=0.895
Epoch: 09 | Time: 0m 42s
	Train Loss: 2.534 | Train PPL:  12.600
	 Val. Loss: 2.951 |  Val. PPL:  19.127
	 BLEU Score: 17.77
	 BERT Score: Precision=0.897, Recall=0.897, F1 Score=0.897
Epoch: 10 | Time: 0m 42s
	Train Loss: 2.318 | Train PPL:  10.151
	 Val. Loss: 2.964 |  Val. PPL:  19.373
	 BLEU Score: 18.90
	 BERT Score: Precision=0.900, Recall=0.900, F1 Score=0.900
```

