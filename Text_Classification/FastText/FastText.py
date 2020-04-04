import tensorflow as tf

class FastText(object):
    def __init__(self, seq_length, num_classes, vocab_size, embedding_size, l2_parm, lr, decay_rate, decay_step):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.l2_parm = l2_parm
        self.add_placeholder()
        self.add_variable()
        self.operation()
        self.loss_op()
        self.train()
        self.predicion()
        

    def add_placeholder(self):
        self.input_x = tf.placeholder(dtype = tf.int32, shape = [None, self.seq_length], name = 'input_x')
        self.input_y = tf.placeholder(dtype = tf.int32, shape = [None, self.num_classes], name = 'input_y')
        self.dropout = tf.placeholder(dtype = tf.float32, name = 'dropout')

    def add_variable(self):
        self.embedding = tf.Variable(initial_value = tf.random_normal(shape = [self.vocab_size, self.embedding_size]), name = 'embedding_layer')
        self.global_step = tf.Variable(0, name = 'global_step')      

    def operation(self):
        with tf.name_scope('embedding'):
            self.vectors = tf.nn.embedding_lookup(self.embedding, ids = self.input_x)
            self.average_vectors = tf.reduce_mean(self.vectors, axis = 1)
            #self.average_vectors = tf.nn.dropout(self.average_vectors, self.dropout)


        with tf.name_scope('linear'):
            self.logits = tf.layers.dense(inputs = self.average_vectors, units= self.num_classes, kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_parm))

    def loss_op(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y))

    def train(self):
        with tf.name_scope('train'):
            learning_rate = tf.train.exponential_decay(self.lr, global_step = self.global_step, decay_rate = self.decay_rate, decay_steps = self.decay_step, staircase = True)
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def predicion(self):
        with tf.name_scope('acc'):
            pred = tf.cast(tf.argmax(self.logits, 1), dtype=tf.int32)
            labels = tf.cast(tf.argmax(self.input_y, 1), dtype=tf.int32)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32), name = 'accuracy')
