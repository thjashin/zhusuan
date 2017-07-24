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
import zhusuan as zs

from examples.utils import dataset
from examples import conf

# from tensorflow.python import debug as tf_debug


class BayesianLSTMCell(object):
    def __init__(self, n_particles, num_units, forget_bias=1.0):
        self._n_particles = n_particles
        self._num_units = num_units
        self._forget_bias = forget_bias
        w_mean = tf.zeros([1, 4 * self._num_units, 2 * self._num_units + 1])
        self._w = zs.Normal(
            'w', w_mean, std=1., n_samples=self._n_particles,
            group_event_ndims=2)

    @property
    def n_particles(self):
        return self._n_particles

    def __call__(self, state, inputs):
        # c: [n_particles, batch_size, num_units],
        # h: [n_particles, batch_size, num_units]
        c, h = state
        batch_size = tf.shape(inputs)[0]
        # inputs: [batch_size, input_size]
        # concat: [n_particles, batch_size, input_size + num_units]
        concat = tf.concat(
            [tf.tile(tf.expand_dims(inputs, 0), [self._n_particles, 1, 1]), h],
            axis=2)
        concat = tf.concat(
            [concat, tf.ones([self._n_particles, batch_size, 1])], axis=2)
        linear_in = tf.expand_dims(concat, -1)
        # w: [n_particles, 1, 4 * num_unit, input_size + num_units + 1]
        # linear_in: [n_particles, batch_size, input_size + num_units + 1, 1]
        linear_out = tf.matmul(tf.tile(self._w, [1, batch_size, 1, 1]),
                               linear_in)
        # linear_out: [n_particles, batch_size, 4 * num_units]
        linear_out = tf.squeeze(linear_out, -1)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=linear_out, num_or_size_splits=4, axis=2)

        new_c = (c * tf.sigmoid(f + self._forget_bias) +
                 tf.sigmoid(i) * tf.tanh(j))
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        return new_c, new_h


def bayesian_rnn(config, cell, input_data, seq_len):
    vocab_size = config.vocab_size
    batch_size = tf.shape(input_data)[0]

    embedding = tf.get_variable("embedding", shape=[vocab_size, 128],
                                dtype=tf.float32)
    # transpose to time major: [max_time, batch_size]
    idx = tf.transpose(input_data)
    # inputs: [max_time, batch_size, 128]
    inputs = tf.nn.embedding_lookup(embedding, idx)
    initializer = (tf.zeros([cell.n_particles, batch_size, 128]),
                   tf.zeros([cell.n_particles, batch_size, 128]))
    c_list, h_list = tf.scan(cell, inputs, initializer=initializer)
    # outputs: [max_time, n_particles, batch_size, 128]
    outputs = h_list
    # outputs: [max_time, batch_size, n_particles, 128]
    outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
    # relevant_outputs = [batch_size, n_particles, 128]
    relevant_outputs = tf.gather_nd(
        outputs, tf.stack([seq_len - 1, tf.range(batch_size)], axis=1))
    # relevant_outputs = [n_particles, batch_size, 128]
    relevant_outputs = tf.transpose(relevant_outputs, [1, 0, 2])
    # logits: [n_particles, batch_size]
    logits = tf.squeeze(tf.layers.dense(relevant_outputs, 1), -1)
    return logits


class Model:
    def __init__(self, N, config):
        # x: [batch_size, seq_len]
        self.x = tf.placeholder(tf.int32, [None, None])
        # y: [batch_size,]
        self.y = tf.placeholder(tf.float32, [None])
        # seq_len: [batch_size,]
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.n_particles = tf.placeholder(tf.int32, [])
        self.config = config

        @zs.reuse("model")
        def build_model(observed):
            with zs.BayesianNet(observed) as model:
                cell = BayesianLSTMCell(self.n_particles, 128, forget_bias=0.)
                logits = bayesian_rnn(self.config, cell, self.x, self.seq_len)
                _ = zs.Bernoulli('y', logits, dtype=tf.float32)
            return model, logits

        @zs.reuse("variational")
        def build_variational():
            with zs.BayesianNet() as variational:
                w_mean = tf.get_variable(
                    'w_mean', shape=[1, 4 * 128, 128 * 2 + 1],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable(
                    'w_logstd', shape=[1, 4 * 128, 128 * 2 + 1],
                    initializer=tf.constant_initializer(-5.))
                w = zs.Normal('w', w_mean, logstd=w_logstd,
                              n_samples=self.n_particles, group_event_ndims=2)
            return variational

        variational = build_variational()
        qw_samples, log_qw = variational.query('w', outputs=True,
                                               local_log_prob=True)

        def log_joint(observed):
            model, _ = build_model(observed)
            log_pw = model.local_log_prob('w')
            log_py_x = model.local_log_prob('y')
            # shape: [n_particles, batch_size]
            return log_pw + log_py_x * N

        lower_bound = tf.reduce_mean(
            zs.sgvb(log_joint, {'y': self.y}, {'w': [qw_samples, log_qw]},
                    axis=0))
        self.loss = -lower_bound

        model, logits = build_model({'w': qw_samples, 'y': self.y})
        log_py_xw = model.local_log_prob('y')
        self.log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0))

        # p: [batch_size,]
        p = tf.reduce_mean(tf.sigmoid(logits), axis=0)
        y_pred = tf.to_float(tf.greater(p, 0.5))
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.y, y_pred)))


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


def run_epoch(sess, model, data, config, train_op=None, n_particles=1):
    epoch_time = -time.time()
    costs, accs, lls = [], [], []
    x_data, y_data = data
    n_iters = int(np.ceil(len(x_data) / float(config.batch_size)))
    for i in range(n_iters):
        x_batch = x_data[i * config.batch_size:(i + 1) * config.batch_size]
        x_batch, x_len = pad_sequences(x_batch, config.max_len)
        feed_dict = {
            model.x: x_batch,
            model.seq_len: x_len,
            model.y: y_data[i * config.batch_size:(i + 1) * config.batch_size],
            model.n_particles: n_particles
        }
        if train_op:
            _, cost = sess.run(
                [train_op, model.loss], feed_dict=feed_dict)
            costs.append(cost)
        else:
            ll, acc = sess.run([model.log_likelihood, model.accuracy],
                               feed_dict=feed_dict)
            lls.append(ll)
            accs.append(acc)

        # print("Iter {}: cost = {}, accuracy = {}".format(i, cost, acc))
        if i % (n_iters // 10) == 10:
            cost_or_ll = np.mean(costs) if costs else np.mean(lls)
            print("%.3f cost/ll: %0.3f acc: %0.4f" %
                  (i * 1.0 / n_iters, cost_or_ll, np.mean(accs)))

    epoch_time += time.time()
    return np.mean(costs) if costs else np.mean(lls), np.mean(accs), epoch_time


def main():
    tf.set_random_seed(1234)

    class SmallConfig(object):
        """Small config."""
        init_scale = 0.1
        learning_rate = 0.001
        max_grad_norm = 5
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
    N =  len(x_train)
    print('train:', len(x_train))
    print('test:', len(x_test))

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("Model", initializer=initializer):
        model = Model(N, config)

    learning_rate = tf.Variable(0.0, trainable=False)
    new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
    update_lr = tf.assign(learning_rate, new_lr)
    var_list = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(model.loss, var_list), config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, var_list))

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

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

            valid_ll, valid_acc, valid_time = run_epoch(
                sess, model, (x_test, y_test), config, n_particles=10)
            print('Epoch {} ({:.1f}s): valid ll = {}, accuracy = {}'
                  .format(epoch, valid_time, valid_ll, valid_acc))


if __name__ == "__main__":
    main()
