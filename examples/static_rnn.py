#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import time

import tensorflow as tf
from tensorflow.contrib import seq2seq
from six.moves import range, zip
import numpy as np

from examples import reader


tf.flags.DEFINE_string("data_path", "data/ptb",
                       "Where the training/test data is stored.")
FLAGS = tf.flags.FLAGS

# from tensorflow.python import debug as tf_debug


def rnn(config, input_data):
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size

    def cell():
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            hidden_size, forget_bias=0., reuse=tf.get_variable_scope().reuse)
        return tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=config.keep_prob)

    multi_cell = tf.nn.rnn_cell.MultiRNNCell(
        [cell() for _ in range(config.num_layers)])
    embedding = tf.get_variable("embedding", shape=[vocab_size, hidden_size],
                                dtype=tf.float32)
    # inputs: [batch_size, num_steps, hidden_size]
    inputs = tf.nn.embedding_lookup(embedding, input_data)
    initial_state = multi_cell.zero_state(batch_size, tf.float32)

    outputs, state = tf.nn.static_rnn(multi_cell, tf.unstack(inputs, axis=1),
                                      dtype=tf.float32)
    outputs = tf.stack(outputs, axis=1)
    # logits: [batch_size, num_steps, vocab_size]
    logits = tf.layers.dense(outputs, vocab_size)
    return logits, initial_state, state


class Model:
    def __init__(self, data, config, name=None):
        self.input, self.targets = reader.ptb_producer(
            data, config.batch_size, config.num_steps, name=name)
        self.n_iters = (len(data) // config.batch_size - 1) // config.num_steps

        self.logits, self.initial_state, self.final_state = rnn(
            config, self.input)
        loss = seq2seq.sequence_loss(
            self.logits,
            self.targets,
            tf.ones([config.batch_size, config.num_steps]),
            average_across_timesteps=False,
            average_across_batch=True
        )
        self.loss = tf.reduce_sum(loss)


def run_epoch(sess, model, config, train_op=None):
    epoch_time = -time.time()
    state = sess.run(model.initial_state)
    costs = []
    for i in range(model.n_iters):
        feed_dict = {}
        for j, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[j].c
            feed_dict[h] = state[j].h

        if train_op:
            _, cost, state = sess.run(
                [train_op, model.loss, model.final_state], feed_dict=feed_dict)
        else:
            cost, state = sess.run([model.loss, model.final_state],
                                   feed_dict=feed_dict)
        print("Iter {}: {}".format(i, cost))
        costs.append(cost)
        if i % (model.n_iters // 10) == 10:
            print("%.3f perplexity: %.3f" %
                  (i * 1.0 / model.n_iters,
                   np.exp(np.mean(costs) / config.num_steps)))
    perplexity = np.exp(np.mean(costs) / config.num_steps)
    epoch_time += time.time()
    return perplexity, epoch_time


def main():
    raw_data = reader.ptb_raw_data(data_path=FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data
    print('train_data:', len(train_data))

    class SmallConfig(object):
        """Small config."""
        init_scale = 0.1
        learning_rate = 1.0
        max_grad_norm = 5
        num_layers = 2
        num_steps = 20
        hidden_size = 200
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 20
        vocab_size = 10000

    # train_input/targets: [batch_size, num_steps]
    config = SmallConfig()
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", initializer=initializer):
            train_model = Model(train_data, config, name="TrainInput")

        learning_rate = tf.Variable(0.0, trainable=False)
        new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
        update_lr = tf.assign(learning_rate, new_lr)
        var_list = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(train_model.loss, var_list), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, var_list))

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_model = Model(valid_data, config, name="ValidInput")

    eval_config = SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            test_model = Model(test_data, eval_config, name="TestInput")

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(1, config.max_max_epoch + 1):
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
            sess.run(update_lr,
                     feed_dict={new_lr: config.learning_rate * lr_decay})
            ppl, train_time = run_epoch(sess, train_model, config,
                                        train_op=train_op)
            print('Epoch {} ({:.1f}s): train perplexity = {}'
                  .format(epoch, train_time, ppl))

            valid_ppl, valid_time = run_epoch(sess, valid_model, config)
            print('Epoch {} ({:.1f}s): valid perplexity = {}'
                  .format(epoch, valid_time, valid_ppl))

        test_ppl, test_time = run_epoch(sess, test_model, eval_config)
        print('Epoch {} ({:.1f}s): test perplexity = {}'
              .format(epoch, test_time, test_ppl))

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()
