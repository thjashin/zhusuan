#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import time
import os

import tensorflow as tf
from six.moves import range, zip
import numpy as np

from examples.utils import dataset
from examples import conf

# from tensorflow.python import debug as tf_debug


class LSTMCell(object):
    def __init__(self, num_units, forget_bias=1.0):
        self._num_units = num_units
        self._forget_bias = forget_bias

    def __call__(self, state, inputs):
        # c: [batch_size, num_units],
        # h: [batch_size, num_units]
        c, h = state
        # inputs: [batch_size, input_size]
        # concat: [batch_size, input_size + num_units]
        concat = tf.concat([inputs, h], axis=1)
        # linear_out: [batch_size, 4 * num_units]
        linear_out = tf.layers.dense(concat, 4 * self._num_units)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=linear_out, num_or_size_splits=4, axis=1)

        new_c = (c * tf.sigmoid(f + self._forget_bias) +
                 tf.sigmoid(i) * tf.tanh(j))
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        return new_c, new_h


def rnn(config, cell, input_data, seq_len):
    vocab_size = config.vocab_size
    batch_size = tf.shape(input_data)[0]

    embedding = tf.get_variable("embedding", shape=[vocab_size, 128],
                                dtype=tf.float32)
    # transpose to time major: [max_time, batch_size]
    idx = tf.transpose(input_data)
    # inputs: [max_time, batch_size, 128]
    inputs = tf.nn.embedding_lookup(embedding, idx)
    initializer = (tf.zeros([batch_size, 128]), tf.zeros([batch_size, 128]))
    c_list, h_list = tf.scan(cell, inputs, initializer=initializer)
    # outputs: [max_time, batch_size, 128]
    outputs = h_list
    # relevant_outputs = [batch_size, 128]
    relevant_outputs = tf.gather_nd(
        outputs, tf.stack([seq_len - 1, tf.range(batch_size)], axis=1))
    # logits: [batch_size,]
    logits = tf.squeeze(tf.layers.dense(relevant_outputs, 1), -1)
    return logits


class Model:
    def __init__(self, x, y, seq_len, config):
        self.x = x
        self.y = y
        self.seq_len = seq_len
        cell = LSTMCell(128, forget_bias=0.)
        self.logits = rnn(config, cell, x, seq_len)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                       logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.accuracy = tf.reduce_mean(
            tf.to_float(tf.equal(y, tf.to_float(tf.less(0., self.logits)))))


def pad_sequences(seqs, max_len=None):
    if max_len:
        seqs = [seq[-max_len:] for seq in seqs]
    max_len = max(len(seq) for seq in seqs)
    padded_seqs = np.zeros((len(seqs), max_len), dtype=np.int32)
    seq_len = []
    for i, seq in enumerate(seqs):
        padded_seqs[i, :len(seq)] = seq
        seq_len.append(len(seq))
    return padded_seqs, np.array(seq_len, dtype=np.int32)


def run_epoch(sess, model, data, config, train_op=None):
    epoch_time = -time.time()
    costs, accs = [], []
    x_data, y_data = data
    n_iters = int(np.ceil(len(x_data) / float(config.batch_size)))
    for i in range(n_iters):
        x_batch = x_data[i * config.batch_size:(i + 1) * config.batch_size]
        x_batch, x_len = pad_sequences(x_batch, config.max_len)
        feed_dict = {
            model.x: x_batch,
            model.seq_len: x_len,
            model.y: y_data[i * config.batch_size:(i + 1) * config.batch_size]
        }
        if train_op:
            _, cost, acc = sess.run(
                [train_op, model.loss, model.accuracy], feed_dict=feed_dict)
        else:
            cost, acc = sess.run([model.loss, model.accuracy],
                                 feed_dict=feed_dict)
        # print("Iter {}: cost = {}, accuracy = {}".format(i, cost, acc))
        costs.append(cost)
        accs.append(acc)
        if i % (n_iters // 10) == 10:
            print("%.3f cost: %0.3f acc: %0.4f" %
                  (i * 1.0 / n_iters, np.mean(costs), np.mean(accs)))
    epoch_time += time.time()
    return np.mean(costs), np.mean(accs), epoch_time


def main():
    class SmallConfig(object):
        """Small config."""
        init_scale = 0.1
        learning_rate = 0.001
        max_grad_norm = 5
        num_layers = 2
        num_steps = 20
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 32
        vocab_size = 20000
        max_len = 80

    config = SmallConfig()
    data_path = os.path.join(conf.data_dir, 'imdb.npz')
    (x_train, y_train), (x_test, y_test) = dataset.imdb_raw_data(
        data_path, num_words=config.vocab_size)
    print('train:', len(x_train))
    print('test:', len(x_test))

    # x: [batch_size, seq_len]
    x = tf.placeholder(tf.int32, [None, None])
    # y: [batch_size,]
    y = tf.placeholder(tf.float32, [None])
    # seq_len: [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("Model", initializer=initializer):
        model = Model(x, y, seq_len, config)

    learning_rate = tf.Variable(0.0, trainable=False)
    new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
    update_lr = tf.assign(learning_rate, new_lr)
    var_list = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(model.loss, var_list), config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, var_list))

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, config.max_max_epoch + 1):
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
            sess.run(update_lr,
                     feed_dict={new_lr: config.learning_rate * lr_decay})
            cost, acc, train_time = run_epoch(
                sess, model, (x_train, y_train), config, train_op=train_op)
            print('Epoch {} ({:.1f}s): train cost = {}, accuracy = {}'
                  .format(epoch, train_time, cost, acc))

            valid_cost, valid_acc, valid_time = run_epoch(
                sess, model, (x_test, y_test), config)
            print('Epoch {} ({:.1f}s): valid cost = {}, accuracy = {}'
                  .format(epoch, valid_time, valid_cost, valid_acc))


if __name__ == "__main__":
    main()
