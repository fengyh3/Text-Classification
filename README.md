# Text-Classification
Deep Learning for Text Classification in NLP.

Enviroment: py3 + tensorflow 1.4+
Movie Review dataset is from [this website](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

Now it contain four models: CNN/BiLSTM/BiLSTM+attention/FastText.(To be continued...)

Some results are in below:

|      | CNN    | BiLSTM    | BiLSTM + attention | FastText |
| ---- | ------ | ------ | ------ | ---------- |
|accuracy   | 76.2% | 79.5% | 76.9% | 80.3%     |

Note that the models do not contain save and load model in tensorflow, and it contains visulazation using tensorboard. Moreover, the models just simply ajust the hyper-parameters and in FastText it just uses unigram. So it just a toy-level demo and use it to learn the text classification.

Now it will be continued with HAN, BERT and so on.
