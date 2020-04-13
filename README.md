# Text-Classification
Deep Learning for Text Classification in NLP.

# Enviroment
py3 + tensorflow 1.12+

# Dataset
Movie Review dataset is from [this website](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

Yelp: it's from [yelp academic review](https://www.kaggle.com/yelp-dataset/yelp-dataset/version/2), i just use first 500,000 texts to train.

# Models
Now it contain four models: CNN/BiLSTM/BiLSTM+attention/FastText/HAN.(To be continued...)

# Results
Some results about accuracy are in below:

|      | CNN    | BiLSTM    | BiLSTM + attention | FastText | RCNN_max-pooling | RCNN_average-pooling|    HAN    |  Bert-Tiny |
| ---- | ------ | ------ | ------ | ---------- |---------------------|-------------------------|-----------------|------------|
|movie review | 76.2% | 79.5% | 76.9% |   80.3%   |     80.4%          |        80.3%            |      -%    |  77.2%  |
|Yelp | 65.1% | 68.2% | 70.2% |  69.5%    |               |                    |    70.5%      |   |

# Tips
Note that the models do not contain save and load model in tensorflow, and it contains visulazation using tensorboard. Moreover, the models just simply ajust the hyper-parameters and in FastText it just uses unigram. So it just a toy-level demo and use it to learn the text classification.

In moview review dataset, we can see that because of the dataset is a bunch of small-scale and short texts, so the complcated DL methods may be not as good as simpler DL methods or ML methods. What's more, the training cost: RCNN > BiLSTM + attention ≈ BiLSTM > CNN >> FastText. And due to movie review dataset is encoding with 'windows-1252', so in training in bert, it causes the messy code and i can't  get a good enough result.

In Yelp dataset, it is a larger-scale dataset and the texts are longer. Due to the limitation of computed resource, the models' hyper-parameter is not a pretty good setting. 

Now it will be continued with Transformer, BERT and so on.
