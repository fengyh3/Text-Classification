import tensorflow as tf
import numpy as np

class HAN(object):
    def __init__(self, max_seq_num, max_seq_len, vocab_size, 
                num_classes, embedding_size, hidden_size,
                 lr, l2_parm, decay_rate, decay_step, grad_clip):
        self.max_seq_num = max_seq_num
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.l2_parm = l2_parm
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.grad_clip = grad_clip
        self.initializer = tf.random_normal_initializer(stddev = 0.1)
        self.add_placeholder()
        self.add_variable()
        self.word2vec()
        self.sent2vec()
        self.doc2vec()
        self.loss_op()
        self.train()
        self.predicion()

    def add_placeholder(self):
        self.input_x = tf.placeholder(dtype = tf.int32, shape = [None, self.max_seq_num, self.max_seq_len], name = 'input_x')
        self.input_y = tf.placeholder(dtype = tf.int32, shape = [None, self.num_classes], name = 'input_y')
        self.dropout = tf.placeholder(dtype = tf.float32, name = 'dropout')

    def add_variable(self):
        self.global_step = tf.Variable(tf.constant(0), name = 'global_step')
        self.embedding = tf.get_variable(name = 'embedding_layer', shape = [self.vocab_size, self.embedding_size], initializer = self.initializer)

    def getSentenceLen(self, sequences):
        abs_sequences = tf.abs(sequences)
        # after padding data, max is 0
        abs_max_seq = tf.reduce_max(abs_sequences, reduction_indices=2)
        max_seq_sign = tf.sign(abs_max_seq)
        # sum is the real length
        real_len = tf.reduce_sum(max_seq_sign, reduction_indices=1)
        return tf.cast(real_len, tf.int32)

    def BiGRU_Encoder(self, inputs, name):
        #input: shape = [batch_size, max_seq_len, embedding_size]
        #output: shape = [batch_size, max_seq_len, 2 * hidden_size]
        with tf.name_scope(name):
            fw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size, name = name + '_fw_cell')
            bw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size, name = name + '_bw_cell')
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, self.dropout)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, self.dropout)
            seq_lengths = self.getSentenceLen(inputs)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs = inputs, sequence_length = seq_lengths, dtype = tf.float32)
            output = tf.concat(outputs, axis = 2)
            return output, seq_lengths

    def Attention_layer(self, inputs, seq_lengths, name):
        #input shae = [batch_size, max_seq_len, 2 * hidden_size]
        with tf.name_scope(name):
            attention_weight = tf.get_variable(name = name + '_attentionWeight', shape = [self.hidden_size * 2], initializer = self.initializer)
            hidden_representation = tf.layers.dense(inputs, units = 2 * self.hidden_size, activation = tf.nn.tanh, name = name + '_hidden')
            reduce_sum = tf.reduce_sum(tf.multiply(hidden_representation, attention_weight), axis = 2, keepdims = True)

            alpha = tf.nn.softmax(reduce_sum, dim = 1)

            zeros = tf.zeros(shape = tf.shape(alpha))
            mask = tf.sequence_mask(seq_lengths, maxlen = self.max_seq_len)
            s = tf.shape(mask)
            mask = tf.reshape(mask, shape = [s[0], s[1], 1])

            alpha = tf.where(mask, alpha, zeros)
            output = tf.reduce_sum(tf.multiply(inputs, alpha), axis = 1)
            return output


    def word2vec(self):
        with tf.name_scope('word2vec'):
            #shape = [batch_size, max_seq_num, max_seq_len, embedding_size]
            self.word_vectors = tf.nn.embedding_lookup(self.embedding, ids = self.input_x, name = 'word_vectors')

    def sent2vec(self):
        #reshape [batch_size, max_seq_num, max_seq_len, embedding_size] to [batch_size * max_seq_num, max_seq_len, embedding_size]
        #so that batch_size -> batch_size * max_seq_num, it will combine with a sentence's meaning
        with tf.name_scope('sent2vec'):
            sen_vec = tf.reshape(self.word_vectors, shape = [-1, self.max_seq_len, self.embedding_size])
            word_encoder, seq_lengths = self.BiGRU_Encoder(sen_vec, 'word_encoder')
            #shape = [batch_size * max_seq_num, 2 * hidden_size]
            self.sent_vectors = self.Attention_layer(word_encoder, seq_lengths, 'word_attention')

    def doc2vec(self):
        #reshape sent_vectors -> [batch_size, max_seq_num, 2 * hidden_size]
        with tf.name_scope('doc2vec'):
            sent_vec = tf.reshape(self.sent_vectors, shape = [-1, self.max_seq_num, 2 * self.hidden_size])
            sent_encoder, seq_lengths = self.BiGRU_Encoder(sent_vec, 'sent_encoder')
            self.doc_vectors = self.Attention_layer(sent_encoder, seq_lengths, 'sent_attention')

    def loss_op(self):
        with tf.name_scope('loss_op'):
            self.logits = tf.layers.dense(inputs = self.doc_vectors, units= self.num_classes, kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_parm))
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y))

    def train(self):
        with tf.name_scope('train'):
            learning_rate = tf.train.exponential_decay(self.lr, global_step = self.global_step, decay_rate = self.decay_rate, decay_steps = self.decay_step, staircase = True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grad_and_vars = optimizer.compute_gradients(self.loss)
            grad_and_vars = [(tf.clip_by_norm(grad, self.grad_clip), val) for grad, val in grad_and_vars if grad is not None]
            self.train_op = optimizer.apply_gradients(grad_and_vars, global_step = self.global_step)

    def predicion(self):
        with tf.name_scope('predicion'):
            pred = tf.cast(tf.argmax(self.logits, 1), dtype=tf.int32)
            labels = tf.cast(tf.argmax(self.input_y, 1), dtype=tf.int32)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32), name = 'accuracy')
