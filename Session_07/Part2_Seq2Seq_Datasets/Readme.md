# 1. Quora Dataset

The dataset can be found here - https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs

## Data Preprocessing
* Columns in the dataset - `['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']`
* Shape of the dataset - `(404290, 6)`
* Shape of the dataset where duplicates are True - `(149263, 6)`
* Dataset head-

|     | id 	| qid1  |	qid2 |	question1 |	question2 |	is_duplicate|
| --- |:-----:|:-----:|:----:| :--------:|:--------:|------------:|
|0 	  |  5 	  | 11 |	12 	| Astrology: I am a Capricorn Sun Cap moon and c... |	I'm a triple Capricorn (Sun, Moon and ascendan... 	| 1 |
|1 	|  7 	  | 15 |	16| 	How can I be a good geologist? |	What should I do to be a great geologist? |	1|
|2 	  | 11 	  | 23 |	24 |	How do I read and find my YouTube comments? |	How can I see all my Youtube comments? |	1|
|3 	  | 12 	  | 25 |	26 |	What can make Physics easy to learn? |	How can you make physics easy to learn? |	1|
|4 	  | 13 	  | 27 |	28 |	What was your first sexual experience like? |	What was your first sexual experience? 	|1|

*  Example of the dataset: 

`{'src': ['astrology', ':', 'i', 'am', 'a', 'capricorn', 'sun', 'cap', 'moon', 'and', 'cap', 'rising', '...', 'what', 'does', 'that', 'say', 'about', 'me', '?'],`

`'trg': ['i', "'m", 'a', 'triple', 'capricorn', '(', 'sun', ',', 'moon', 'and', 'ascendant', 'in', 'capricorn', ')', 'what', 'does', 'this', 'say', 'about', 'me', '?']}`

## Train Test Split

* Number of training examples: `104484`
* Number of testing examples: `44779`

## Vocab
* Unique tokens in source vocabulary: `23838`
* Unique tokens in target vocabulary: `23884`

# 2. Question Answer Dataset

The dataset can be found here - http://www.cs.cmu.edu/~ark/QA-data/

## Data Preprocessing

* Data file structure in .txt 
```
Filename: Question_Answer_Dataset_v1.2/S08/question_answer_pairs.txt
Shape: (1715, 6)

Filename: Question_Answer_Dataset_v1.2/S09/question_answer_pairs.txt
Shape: (825, 6)

Filename: Question_Answer_Dataset_v1.2/S10/question_answer_pairs.txt
Shape: (1458, 6)
```

* Columns in the dataset - `['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']`

Shape of the dataset - `(404290, 6)`

Shape of the dataset where duplicates are True - `(149263, 6)`

Dataset head-

|     | id 	| qid1  |	qid2 |	question1 |	question2 |	is_duplicate|
| --- |:-----:|:-----:|:----:| :--------:|:--------:|------------:|
|0 	  |  5 	  | 11 |	12 	| Astrology: I am a Capricorn Sun Cap moon and c... |	I'm a triple Capricorn (Sun, Moon and ascendan... 	| 1 |
|1 	|  7 	  | 15 |	16| 	How can I be a good geologist? |	What should I do to be a great geologist? |	1|
|2 	  | 11 	  | 23 |	24 |	How do I read and find my YouTube comments? |	How can I see all my Youtube comments? |	1|
|3 	  | 12 	  | 25 |	26 |	What can make Physics easy to learn? |	How can you make physics easy to learn? |	1|
|4 	  | 13 	  | 27 |	28 |	What was your first sexual experience like? |	What was your first sexual experience? 	|1|

Example of the dataset: 

`{'src': ['astrology', ':', 'i', 'am', 'a', 'capricorn', 'sun', 'cap', 'moon', 'and', 'cap', 'rising', '...', 'what', 'does', 'that', 'say', 'about', 'me', '?'],`

`'trg': ['i', "'m", 'a', 'triple', 'capricorn', '(', 'sun', ',', 'moon', 'and', 'ascendant', 'in', 'capricorn', ')', 'what', 'does', 'this', 'say', 'about', 'me', '?']}`

## Train Test Split

Number of training examples: `104484`

Number of testing examples: `44779`

## Vocab

Unique tokens in source vocabulary: `23838`

Unique tokens in target vocabulary: `23884`


