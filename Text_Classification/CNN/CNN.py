import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(self, seq_length, num_classes, vocab_size, embedding_size, filter_size, num_filters, l2_parm):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.l2_parm = l2_parm
        self.word_embedding = tf.Variable(initial_value = tf.random_normal(shape=[vocab_size, embedding_size]), trainable = True)
        self.add_placeholder()
        self.operation()
        self.loss_op()

    def add_placeholder(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.seq_length], name='x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, self.num_classes], name='y')
        self.droupout = tf.placeholder(tf.float32, name='droupout')

    def convolution(self, filter_size):
        filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
        filter_W = tf.Variable(initial_value=tf.truncated_normal(filter_shape, stddev=0.1), name = 'filter_W')
        filter_b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name = 'filter_b')
        conv = tf.nn.conv2d(input = self.vectors, filter = filter_W, strides = [1, 1, 1, 1], padding = 'VALID', name = 'conv')
        activate = tf.nn.relu(tf.nn.bias_add(conv, filter_b), name = 'relu')
        pooling = tf.nn.max_pool(activate, ksize = [1, self.seq_length - filter_size + 1, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID', name = 'max_pool')
        return pooling


    def operation(self):
        with tf.name_scope('embedding'):
            vectors = tf.nn.embedding_lookup(self.word_embedding, ids = self.input_x)
            #Con2d need 4-dimension vector, shape = [batch, width, height, channels]
            #original paper said that it can use muti-channel which has different embedding.
            self.vectors = tf.expand_dims(vectors, axis = -1, name='vectors')
        pools = []
        for i, filter_size in enumerate(self.filter_size):
            with tf.name_scope('conv-max_pool%s' % filter_size):
                pools.append(self.convolution(filter_size))

        total_num_filters = self.num_filters * len(self.filter_size)
        feature = tf.concat(pools, axis=3)
        feature = tf.reshape(feature, shape=(-1, total_num_filters))

        with tf.name_scope('droup'):
            self.feature = tf.nn.dropout(feature, self.droupout)


    def loss_op(self):
        with tf.name_scope('output'):
            self.logits = tf.layers.dense(inputs = self.feature, units= self.num_classes, kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_parm))

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y))

        with tf.name_scope('acc'):
            pred = tf.cast(tf.argmax(self.logits, 1), dtype=tf.int32)
            labels = tf.cast(tf.argmax(self.input_y, 1), dtype=tf.int32)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32), name = 'accuracy')