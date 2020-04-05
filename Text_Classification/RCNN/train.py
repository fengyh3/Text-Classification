import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
from CNN.data import *
from RCNN import RCNN
import os
import time

#data parameter
tf.flags.DEFINE_float("dev_per", 0.1, "percent of dev dataset")
tf.flags.DEFINE_string("positive_file_name", "../CNN/data/rt-polarity.pos", "positive dataset file name")
tf.flags.DEFINE_string("negetive_file_name", "../CNN/data/rt-polarity.neg", "negetive dataset file name")

#model parameter
tf.flags.DEFINE_integer("embedding_size", 128, "char or word embedding size")
tf.flags.DEFINE_integer("hidden_size", 128, "LSTM hidden size")
tf.flags.DEFINE_integer("context_size", 128, "measure context of a word in RCNN")
tf.flags.DEFINE_integer("decay_step", 12000, "using lr decay after decay_step steps")
tf.flags.DEFINE_float("decay_rate", 0.9, "each time decay decay_rate percent")
tf.flags.DEFINE_float("droupout_pro", 0.8, "droupout probability in LSTM")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("l2_reg", 0.001, "l2 regularization parameter")
tf.flags.DEFINE_float("grad_clip", 5.0, "gradient clip when larger than grad_clip")
tf.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.flags.DEFINE_integer("epoches", 60, "epoches in training")
tf.flags.DEFINE_integer("save_step", 50, "every save_step to save model")
tf.flags.DEFINE_integer("evaluate_step", 10, "every evaluate_step to test the model in dev dataset")
tf.flags.DEFINE_integer("num_checkpoint", 5, "how many checkpoints are saved each time")

FLAGS = tf.flags.FLAGS

def visualize(save_dir, rcnn, sess):
    loss_summary = tf.summary.scalar('loss', rcnn.loss)
    acc_summary = tf.summary.scalar('accuracy', rcnn.acc)

    train_summary = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(save_dir, 'summary', 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(save_dir, 'summary', 'dev')
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    return train_summary, dev_summary, train_summary_writer, dev_summary_writer



def train(x_train, y_train, x_dev, y_dev, vocab, train_seq_len = None, dev_seq_len = None):
    sess = tf.Session()
    with sess.as_default():
        rcnn = RCNN(max_seq_len = x_train.shape[1], num_classes = y_train.shape[1], hidden_size = FLAGS.hidden_size,
                      vocab_size = len(vocab.vocabulary_), embedding_size = FLAGS.embedding_size, 
                      decay_step = FLAGS.decay_step, decay_rate = FLAGS.decay_rate,
                      lr = FLAGS.learning_rate, l2_parm = FLAGS.l2_reg, context_size = FLAGS.context_size, grad_clip = FLAGS.grad_clip)

        global_step = tf.Variable(0, name = 'global_step', trainable = False)


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
        train_sum, dev_sum, train_writer, dev_writer = visualize(save_dir, rcnn, sess)

        sess.run(tf.global_variables_initializer())
        batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.epoches)
        for batch in batches:
            batch_x, batch_y= zip(*batch)
            feed_dict = {
                rcnn.input_x: batch_x,
                rcnn.input_y: batch_y,
                rcnn.dropout: FLAGS.droupout_pro
            }
            _, step, summary, loss, acc = sess.run([rcnn.train_op, global_step, train_sum, rcnn.loss, rcnn.acc], feed_dict = feed_dict)
            print("step %s: loss: %s, acc: %s" % (step, loss, acc))
            train_writer.add_summary(summary, step)
            if step % FLAGS.evaluate_step == 0:
                feed_dict = {
                    rcnn.input_x: x_dev,
                    rcnn.input_y: y_dev,
                    rcnn.dropout: 1.0
                }
                summary, loss, acc = sess.run([dev_sum, rcnn.loss, rcnn.acc], feed_dict = feed_dict)
                print("Dev result: loss: %s, acc: %s" % (loss, acc))
                dev_writer.add_summary(summary, step)
            if step % FLAGS.save_step == 0:
                path = saver.save(sess, checkpoint_prefix, global_step = step)
            global_step = tf.add(global_step, 1)

def main(argv=None):
    x_train, y_train, x_dev, y_dev, vocab, _, _= data_process(FLAGS.positive_file_name, FLAGS.negetive_file_name, FLAGS.dev_per)
    train(x_train, y_train, x_dev, y_dev, vocab)

if __name__ == '__main__':
    tf.app.run()