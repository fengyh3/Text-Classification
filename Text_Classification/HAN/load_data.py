import numpy as np
import json
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict

def pre_process():
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    word_tokenizer = WordPunctTokenizer()
    print('load vocab...')
    with open('../yelp_data/vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
    print('finish vocab')

    data_x = []
    data_y = []
    max_sent_in_doc = 30
    max_word_in_sent = 30
    num_classes = 5

    print('process data...')
    time = 1
    with open('../yelp_data/yelp_academic_dataset_review.json', 'rb') as f:
        for line in f:
            time += 1
            if time >= 500000:
                break
            doc = np.array(max_sent_in_doc * [max_word_in_sent * [0]])
            review = json.loads(line)
            sents = sent_tokenizer.tokenize(review['text'])
            for i, sent in enumerate(sents):
                if i < max_sent_in_doc:
                    for j, word in enumerate(word_tokenizer.tokenize(sent)):
                        if j < max_word_in_sent:
                            doc[i][j] = vocab.get(word, 0)

            label = int(review['stars'])
            labels = np.array([0] * num_classes)
            labels[label - 1] = 1
            data_x.append(doc)
            data_y.append(labels)
        print(len(data_x))
        pickle.dump((data_x, data_y), open('yelp_data', 'wb'))


def read_dataset(dev_per):
    with open('yelp_data', 'rb') as f:
        x, y = pickle.load(f)
    dev_part = int(dev_per * len(x))
    x = np.array(x)
    y = np.array(y)
    x_train = x[dev_part : ]
    y_train = y[dev_part : ]
    x_dev = x[ : dev_part]
    y_dev = y[ : dev_part]
    
    return x_train, y_train, x_dev, y_dev
