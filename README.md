# Text-Classification
Deep Learning for Text Classification in NLP.

Enviroment: py3 + tensorflow 1.4+
Movie Review dataset is from [this website](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

Now it contain four models: CNN/BiLSTM/BiLSTM+attention/FastText.(To be continued...)

Some results are in below:

|      | CNN    | BiLSTM    | BiLSTM + attention | FastText | RCNN_max-pooling | RCNN_average-pooling|
| ---- | ------ | ------ | ------ | ---------- |---------------------|-------------------------|
|accuracy   | 76.2% | 79.5% | 76.9% |   80.3%   |     80.4%          |        80.3%            |    

Note that the models do not contain save and load model in tensorflow, and it contains visulazation using tensorboard. Moreover, the models just simply ajust the hyper-parameters and in FastText it just uses unigram. So it just a toy-level demo and use it to learn the text classification.

We can see that because of the dataset is a bunch of small-scale and short texts, so the complcated DL methods may be not as good as simpler DL methods or ML methods. What's more, the training cost: RCNN > BiLSTM + attention ≈ BiLSTM > CNN >> FastText.

Now it will be continued with HAN, BERT and so on.
