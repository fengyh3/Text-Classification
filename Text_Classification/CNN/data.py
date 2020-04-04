import numpy as np
import re
import tensorflow as tf

def clean_str(string):
    """
    Clean the text data using the same code as the original paper
    from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param string: input string to process
    :return: a sentence with lower representation and delete spaces
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(positive_file_name, negetive_file_name):
    positive = list(open(positive_file_name, 'r', encoding = 'windows-1252').readlines())
    positive = [sent.strip() for sent in positive]
    negetive = list(open(negetive_file_name, 'r', encoding = 'windows-1252').readlines())
    negetive = [sent.strip() for sent in negetive]

    x = positive + negetive
    x = [clean_str(sent) for sent in x]
    positive_labels = [[1, 0] for i in range(len(positive))]
    negetive_labels = [[0, 1] for i in range(len(negetive))]
    y = np.concatenate([positive_labels, negetive_labels], axis = 0)
    return x, y

def data_process(positive_file_name, negetive_file_name, dev_per):
    x, y = load_data(positive_file_name, negetive_file_name)
    max_seq_len = max(len(sent.split(' ')) for sent in x)
    seq_lengths = [len(sent.split(' ')) for sent in x]
    seq_lengths = np.array(seq_lengths)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_seq_len)
    x = np.array(list(vocab_processor.fit_transform(x)))

    #shuffle
    np.random.seed(1)
    shuffle_indices = np.random.permutation(len(x))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    seq_lengths = seq_lengths[shuffle_indices]

    dev_part = int(dev_per * len(x))
    x_train = x[dev_part : ]
    y_train = y[dev_part : ]
    train_seq_lengths = seq_lengths[dev_part : ]
    x_dev = x[ : dev_part]
    y_dev = y[ : dev_part]
    dev_seq_lengths = seq_lengths[ : dev_part]
    del x, y

    return x_train, y_train, x_dev, y_dev, vocab_processor, train_seq_lengths, dev_seq_lengths

def batch_iter(data, batch_size, epoches):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(epoches):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
        for batch in range(num_batches_per_epoch):
            start_index = batch_size * batch
            end_index = min(batch_size * batch + batch_size, data_size)
            yield data[start_index : end_index]