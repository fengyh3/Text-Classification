import tensorflow as tf
import numpy as np
from data import *
from CNN import TextCNN
import os
import time

#data parameter
tf.flags.DEFINE_float("dev_per", 0.1, "percent of dev dataset")
tf.flags.DEFINE_string("positive_file_name", "./data/rt-polarity.pos", "positive dataset file name")
tf.flags.DEFINE_string("negetive_file_name", "./data/rt-polarity.neg", "negetive dataset file name")

#model parameter
tf.flags.DEFINE_integer("embedding_size", 300, "char or word embedding size")
tf.flags.DEFINE_string("diff_filter_size", "3,4,5", "how many words are covered by each filter")
tf.flags.DEFINE_integer("number_of_filters", 100, "numbers of filters for each size")
tf.flags.DEFINE_float("droupout_pro", 0.5, "droupout probability in CNN")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_float("l2_reg", 3, "l2 regularization parameter")
tf.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.flags.DEFINE_integer("epoches", 100, "epoches in training")
tf.flags.DEFINE_integer("save_step", 50, "every save_step to save model")
tf.flags.DEFINE_integer("evaluate_step", 10, "every evaluate_step to test the model in dev dataset")
tf.flags.DEFINE_integer("num_checkpoint", 5, "how many checkpoints are saved each time")

FLAGS = tf.flags.FLAGS

def visualize(grads_and_vars, save_dir, cnn, sess):
    grad_summary = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
            sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
            grad_summary.append(grad_hist_summary)
            grad_summary.append(sparsity_summary)
    grad_summary_merge = tf.summary.merge(grad_summary)

    loss_summary = tf.summary.scalar('loss', cnn.loss)
    acc_summary = tf.summary.scalar('accuracy', cnn.acc)

    train_summary = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(save_dir, 'summary', 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(save_dir, 'summary', 'dev')
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    return train_summary, dev_summary, train_summary_writer, dev_summary_writer



def train(x_train, y_train, x_dev, y_dev, vocab):
    sess = tf.Session()
    with sess.as_default():
        cnn = TextCNN(seq_length = x_train.shape[1], num_classes = y_train.shape[1], 
                      vocab_size = len(vocab.vocabulary_), embedding_size = FLAGS.embedding_size, 
                      filter_size = list(map(int, FLAGS.diff_filter_size.split(','))), 
                      num_filters = FLAGS.number_of_filters, l2_parm = FLAGS.l2_reg)

        #set optimizer
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

        #save file name
        timestamp = str(time.time())
        save_dir = os.path.abspath(os.path.join(os.path.curdir, 'run', timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(save_dir, 'checkpoint'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = FLAGS.num_checkpoint)
        vocab.save(os.path.join(save_dir, 'vocab'))

        #visualize
        train_sum, dev_sum, train_writer, dev_writer = visualize(grads_and_vars, save_dir, cnn, sess)

        sess.run(tf.global_variables_initializer())
        batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.epoches)
        for batch in batches:
            batch_x, batch_y = zip(*batch)
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.input_y: batch_y,
                cnn.droupout: FLAGS.droupout_pro
            }
            _, step, summary, loss, acc = sess.run([train_op, global_step, train_sum, cnn.loss, cnn.acc], feed_dict = feed_dict)
            print("step %s: loss: %s, acc: %s" % (step, loss, acc))
            train_writer.add_summary(summary, step)
            if step % FLAGS.evaluate_step == 0:
                feed_dict = {
                    cnn.input_x: x_dev,
                    cnn.input_y: y_dev,
                    cnn.droupout: 1.0
                }
                summary, loss, acc = sess.run([dev_sum, cnn.loss, cnn.acc], feed_dict = feed_dict)
                print("Dev result: loss: %s, acc: %s" % (loss, acc))
                dev_writer.add_summary(summary, step)
            if step % FLAGS.save_step == 0:
                path = saver.save(sess, checkpoint_prefix, global_step = step)

def main(argv=None):
    x_train, y_train, x_dev, y_dev, vocab, _, _ = data_process(FLAGS.positive_file_name, FLAGS.negetive_file_name, FLAGS.dev_per)
    train(x_train, y_train, x_dev, y_dev, vocab)

if __name__ == '__main__':
    tf.app.run()