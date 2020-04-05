import tensorflow as tf
import numpy as np
import copy

class RCNN(object):
    def __init__(self, max_seq_len, vocab_size, num_classes, embedding_size, hidden_size, context_size, lr, l2_parm, decay_rate, decay_step, grad_clip):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.lr = lr
        self.l2_parm = l2_parm
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.grad_clip = grad_clip
        self.initializer = tf.random_normal_initializer(stddev = 0.1)
        self.add_placeholder()
        self.add_variable()
        self.operation()
        self.loss_op()
        self.train()
        self.predicion()


    def add_placeholder(self):
        self.input_x = tf.placeholder(dtype = tf.int32, shape = [None, self.max_seq_len], name = 'input_x')
        self.input_y = tf.placeholder(dtype = tf.int32, shape = [None, self.num_classes], name = 'input_y')
        self.dropout = tf.placeholder(dtype = tf.float32, name = 'dropout')
        self.batch_size = tf.shape(self.input_x)[0]

    def add_variable(self):
        self.global_step = tf.Variable(tf.constant(0), name = 'global_step')
        self.embedding = tf.get_variable(name = 'embedding_layer', shape = [self.vocab_size, self.embedding_size], initializer = self.initializer)
        #left context weight
        self.W_l = tf.get_variable(name = 'left_context', shape = [self.context_size, self.context_size], initializer = self.initializer)
        #left current word (in the paper, current + left context = next word's left context)
        self.W_sl = tf.get_variable(name = 'cur_left_context', shape = [self.embedding_size, self.context_size], initializer = self.initializer)
        #right context weight
        self.W_r = tf.get_variable(name = 'right_context', shape = [self.context_size, self.context_size], initializer = self.initializer)
        #right current word
        self.W_sr = tf.get_variable(name = 'cur_right_context', shape = [self.embedding_size, self.context_size], initializer = self.initializer)

    def get_left_context(self, left_context, left_embedding):
        left_c = tf.matmul(left_context, self.W_l)
        left_e = tf.matmul(left_embedding, self.W_sl)

        #if use relu, gradient explosion will be happend, and it may combine a clip the value
        left_context = tf.nn.tanh(left_c + left_e)
        return left_context

    def get_right_context(self, right_context, right_embedding):
        right_c = tf.matmul(right_context, self.W_r)
        right_e = tf.matmul(right_embedding, self.W_sr)

        right_context = tf.nn.tanh(right_c + right_e)
        return right_context

    def RCNN_layer(self):
        # first, split the input to shape = [max_seq_len, batch_size, 1, embedding_size]
        split_data = tf.split(self.vectors, self.max_seq_len, axis = 1)

        #remove 1-dim in split_data, shape = [max_seq_len, batch_size, embedding_size]
        squeeze_data = [tf.squeeze(x, axis = 1) for x in split_data]

        #default start context and embedding vectors -> 0 vector
        previous_context = tf.zeros(shape = [self.batch_size, self.context_size])
        previous_embedding = tf.zeros(shape = [self.batch_size, self.embedding_size])

        left_context_list = []
        for idx, cur_embedding in enumerate(squeeze_data):
            left_context = self.get_left_context(previous_context, previous_embedding)
            left_context_list.append(left_context)
            previous_context = left_context
            previous_embedding = cur_embedding

        squeeze_data_reverse = copy.copy(squeeze_data)
        squeeze_data_reverse.reverse()
        back_context = tf.zeros(shape = [self.batch_size, self.context_size])
        back_embedding = tf.zeros(shape = [self.batch_size, self.embedding_size])

        right_context_list = []
        for idx, cur_embedding in enumerate(squeeze_data_reverse):
            right_context = self.get_right_context(back_context, back_embedding)
            right_context_list.append(right_context)
            back_context = right_context
            back_embedding = cur_embedding

        count = len(right_context_list)
        #shape = [max_seq_len, batch_size, 2 * context_size + embedding_size]
        context_output = []
        for idx, cur_embedding in enumerate(squeeze_data):
            #shape = [batch_size, 2 * context_size + embedding_Size]
            context_output.append(tf.concat([left_context_list[idx], cur_embedding, right_context_list[count - 1 - idx]], axis = 1))

        #shape = [batch_size, max_seq_len, 2 * context_size + embedding_size]
        outputs = tf.stack(context_output, axis = 1)
        return outputs

    def operation(self):
        with tf.name_scope('embedding'):
            self.vectors = tf.nn.embedding_lookup(self.embedding, ids = self.input_x, name = 'vectors')

        with tf.name_scope('RCNN_op'): 
            context_representation = self.RCNN_layer()
            #shape = [batch_size, 2 * context_size + embedding_size]
            #max pooling:
            #pooling = tf.reduce_max(context_representation, axis = 1)

            #average pooling:
            pooling = tf.reduce_mean(context_representation, axis = 1)

        with tf.name_scope('dropout'):
            pooling_drop = tf.nn.dropout(pooling, keep_prob = self.dropout)

        with tf.name_scope('projection'):
            self.logits = tf.layers.dense(inputs = pooling_drop, units= self.num_classes, kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_parm))

    def loss_op(self):
        with tf.name_scope('loss_op'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y))

    def train(self):
        with tf.name_scope('train'):
            learning_rate = tf.train.exponential_decay(self.lr, global_step = self.global_step, decay_rate = self.decay_rate, decay_steps = self.decay_step, staircase = True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grad_and_vars = optimizer.compute_gradients(self.loss)
            #can choose to gradient clip, but in this dataset, the gradient explosion didn't happen
            grad_and_vars = [(tf.clip_by_norm(grad, self.grad_clip), val) for grad, val in grad_and_vars if grad is not None]
            self.train_op = optimizer.apply_gradients(grad_and_vars, global_step = self.global_step)

    def predicion(self):
        with tf.name_scope('predicion'):
            pred = tf.cast(tf.argmax(self.logits, 1), dtype=tf.int32)
            labels = tf.cast(tf.argmax(self.input_y, 1), dtype=tf.int32)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32), name = 'accuracy')