import tensorflow as tf

class LSTM(object):
    def __init__(self, max_seq_len, vocab_size, num_classes, embedding_size, hidden_size, lr, l2_parm, decay_rate, decay_step, grad_clip, attention = True):
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
        self.attention = attention
        self.initializer = tf.random_normal_initializer()
        self.add_placeholder()
        self.add_variable()
        self.operation()
        self.train()
        self.predicion()

    def add_placeholder(self):
        self.input_x = tf.placeholder(dtype = tf.int32, shape = [None, self.max_seq_len], name = 'input_x')
        self.input_y = tf.placeholder(dtype = tf.int32, shape = [None, self.num_classes], name = 'input_y')
        self.seq_lengths = tf.placeholder(dtype = tf.int32, shape = [None], name = 'seq_lengths')
        self.dropout = tf.placeholder(dtype = tf.float32, name = 'dropout')

    def add_variable(self):
        self.global_step = tf.Variable(tf.constant(0), name = 'global_step')
        self.embedding = tf.get_variable(name = 'embedding', shape = [self.vocab_size, self.embedding_size], initializer = self.initializer)

    def attention_layer(self, inputs):
        attention_weight = tf.get_variable(name = 'attention_weight', shape = [2 * self.hidden_size], initializer = self.initializer)
        hidden_representation = tf.layers.dense(inputs, units = 2 * self.hidden_size, activation = tf.nn.tanh)
        reduce_sum = tf.reduce_sum(tf.multiply(hidden_representation, attention_weight), axis = 2, keepdims = True)

        alpha = tf.nn.softmax(reduce_sum, dim = 1)

        zeros = tf.zeros(shape = tf.shape(alpha))
        mask = tf.sequence_mask(self.seq_lengths, maxlen = self.max_seq_len)
        s = tf.shape(mask)
        mask = tf.reshape(mask, shape = [s[0], s[1], 1])

        self.alpha = tf.where(mask, alpha, zeros)
        output = tf.reduce_sum(tf.multiply(inputs, alpha), axis = 1)
        return output

    def operation(self):
        with tf.name_scope('embedding'):
            self.vectors = tf.nn.embedding_lookup(self.embedding, ids = self.input_x, name = 'vectors')

        with tf.name_scope('Bi-LSTM'):
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name = 'fw_lstm_cell')
            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name = 'bw_lstm_cell')
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, inputs = self.vectors, sequence_length = self.seq_lengths, dtype = tf.float32)
            output = tf.concat(outputs, axis = 2)

            if self.attention is True:
                self.output = tf.reduce_mean(output, axis = 1) + self.attention_layer(output)
            else:
                self.output = tf.reduce_mean(output, axis = 1)
            
        with tf.name_scope('projection_layer'):
            self.logits = tf.layers.dense(inputs = self.output, units= self.num_classes, kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_parm))

        with tf.name_scope('loss_op'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y))

    def train(self):
        with tf.name_scope('train'):
            learning_rate = tf.train.exponential_decay(self.lr, global_step = self.global_step, decay_rate = self.decay_rate, decay_steps = self.decay_step, staircase = True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grad_and_vars = optimizer.compute_gradients(self.loss)
            #can choose to gradient clip, but in this dataset, the gradient explosion didn't happen
            #grad_and_vars = [(tf.clip_by_norm(grad, self.grad_clip), val) for grad, val in grad_and_vars if grad is not None]
            self.train_op = optimizer.apply_gradients(grad_and_vars, global_step = self.global_step)


    def predicion(self):
        with tf.name_scope('predicion'):
            pred = tf.cast(tf.argmax(self.logits, 1), dtype=tf.int32)
            labels = tf.cast(tf.argmax(self.input_y, 1), dtype=tf.int32)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32), name = 'accuracy')
