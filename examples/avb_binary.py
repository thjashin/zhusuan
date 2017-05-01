#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range, zip
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs

import dropout_dataset as dataset

import pdb
import logging
from datetime import datetime

def bayesianNN(observed, x, n_x, layer_sizes, n_particles, keep_rate):
    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([n_particles, 1, n_out, n_in + 1])
            w_logstd = tf.zeros([n_particles, 1, n_out, n_in + 1])
            ws.append(zs.Normal('w' + str(i), w_mu, w_logstd))
        # forward
        ly_x = tf.expand_dims(
            tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        for i in range(len(ws)):
            w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat(
                [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.cast(tf.shape(ly_x)[2],
                                                        tf.float32))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        logits = tf.squeeze(ly_x, [3])
        y_class = zs.Discrete('y', logits)
    return model, logits

@zs.reuse('variational')
def mean_field_variational(layer_sizes, n_particles, is_training):
    with zs.BayesianNet() as variational:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mean = tf.get_variable(
                'w_mean_' + str(i), shape=[1, n_out, n_in + 1],
                initializer=tf.constant_initializer(0.))
            w_logstd = tf.get_variable(
                'w_logstd_' + str(i), shape=[1, n_out, n_in + 1],
                initializer=tf.constant_initializer(0.))
            ws.append(
                zs.Normal('w' + str(i), w_mean, w_logstd,
                          n_samples=n_particles, group_event_ndims=2))
    return ws[0], ws[1]

@zs.reuse('variational')
def fully_connected_variational(layer_sizes, n_particles, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}

    w0 = tf.random_normal([n_particles, 100])
    w0 = layers.fully_connected(w0, 1024)
    N = layer_sizes[1] * (layer_sizes[0] + 1)
    w0 = layers.fully_connected(w0, N)
    w0 = layers.fully_connected(w0, N, activation_fn=None)
    w0 = tf.reshape(w0, [n_particles, 1, layer_sizes[1], layer_sizes[0]+1])

    w1 = tf.random_normal([n_particles, 50])
    w1 = layers.fully_connected(w1, 100)
    w1 = layers.fully_connected(w1, 51)
    w1 = layers.fully_connected(w1, 51, activation_fn=None)
    w1 = tf.reshape(w1, [n_particles, 1, 1, 51])
    return w0, w1


@zs.reuse('variational')
def deconv_variational(layer_sizes, n_particles, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}

    w0 = tf.random_normal([n_particles, 500])
    w0 = layers.fully_connected(w0, 1120)
    w0 = tf.reshape(w0, [n_particles, 10, 14, 8])
    w0 = layers.conv2d_transpose(w0, 4, 5, stride=[5, 8])
    w0 = layers.conv2d_transpose(w0, 2, 5, stride=[3, 7])
    w0 = layers.conv2d_transpose(w0, 1, [1, 2], stride=1,
                                  padding='VALID', activation_fn=None)
#    [n_particles, 1, 150, 785]
    w0 = tf.transpose(w0, [0, 3, 1, 2])

    w1 = tf.random_normal([n_particles, 800])
    w1 = layers.fully_connected(w1, 800)
    w1 = tf.reshape(w1, [n_particles, 10, 10, 8])
    w1 = layers.conv2d_transpose(w1, 4, 5, stride=5)
    w1 = layers.conv2d_transpose(w1, 1, 5, stride=3)
    w1 = layers.conv2d_transpose(w1, 1, [1, 2], stride=1,
                                  padding='VALID', activation_fn=None)
#   [n_particles, 1, 150, 151]
    w1 = tf.transpose(w1, [0, 3, 1, 2])

    w2 = tf.random_normal([n_particles, 800])
    w2 = layers.fully_connected(w2, 800)
    w2 = tf.reshape(w2, [n_particles, 10, 10, 8])
    w2 = layers.conv2d_transpose(w2, 4, 5, stride=5)
    w2 = layers.conv2d_transpose(w2, 1, 5, stride=3)
    w2 = layers.conv2d_transpose(w2, 1, [1, 2], stride=1,
                                  padding='VALID', activation_fn=None)
#   [n_particles, 1, 150, 151]
    w2 = tf.transpose(w2, [0, 3, 1, 2])

    w3 = tf.random_normal([n_particles, 100])
    w3 = layers.fully_connected(w3, 320)
    w3 = tf.reshape(w3, [n_particles, 2, 10, 16])
    w3 = layers.conv2d_transpose(w3, 4, 5, stride=[1, 3])
    w3 = layers.conv2d_transpose(w3, 1, 5, stride=[5, 5])
    w3 = layers.conv2d_transpose(w3, 1, [1, 2], stride=1,
                                  padding='VALID', activation_fn=None)
#   [n_particles, 1, 10, 151]
    w3 = tf.transpose(w3, [0, 3, 1, 2])

    return [w0, w1, w2, w3]

@zs.reuse('discriminator')
def fully_connected_discriminator(observed, latent):
    return lc_w0, lc_w1


@zs.reuse('discriminator')
def conv_discriminator(observed, latent, n_particles):
    w0 = tf.transpose(latent['w0'], [0, 2, 3, 1])
    w1 = tf.transpose(latent['w1'], [0, 2, 3, 1])
    w2 = tf.transpose(latent['w2'], [0, 2, 3, 1])
    w3 = tf.transpose(latent['w3'], [0, 2, 3, 1])

    lc_w0 = layers.conv2d(w0, 1, [1, 2], stride=1,
                       padding='VALID')
    lc_w0 = layers.conv2d(lc_w0, 2, 5, stride=[3, 7])
    lc_w0 = layers.conv2d(lc_w0, 4, 5, stride=[5, 8])
    lc_w0 = tf.reshape(lc_w0, [n_particles, 560])
    lc_w0 = layers.fully_connected(lc_w0, 500)
    lc_w0 = layers.fully_connected(lc_w0, 1, activation_fn=None)
    lc_w0 = tf.squeeze(lc_w0, [1])

    lc_w1 = layers.conv2d(w1, 1, [1, 2], stride=1,
                       padding='VALID')
    lc_w1 = layers.conv2d(lc_w1, 2, 5, stride=3)
    lc_w1 = layers.conv2d(lc_w1, 4, 5, stride=5)
    lc_w1 = tf.reshape(lc_w1, [n_particles, 400])
    lc_w1 = layers.fully_connected(lc_w1, 500)
    lc_w1 = layers.fully_connected(lc_w1, 1, activation_fn=None)
    lc_w1 = tf.squeeze(lc_w1, [1])

    lc_w2 = layers.conv2d(w2, 1, [1, 2], stride=1,
                       padding='VALID')
    lc_w2 = layers.conv2d(lc_w2, 2, 5, stride=3)
    lc_w2 = layers.conv2d(lc_w2, 4, 5, stride=5)
    lc_w2 = tf.reshape(lc_w2, [n_particles, 400])
    lc_w2 = layers.fully_connected(lc_w2, 500)
    lc_w2 = layers.fully_connected(lc_w2, 1, activation_fn=None)
    lc_w2 = tf.squeeze(lc_w2, [1])

    lc_w3 = layers.flatten(w3)
    lc_w3 = layers.fully_connected(lc_w3, 100)
    lc_w3 = layers.fully_connected(lc_w3, 100)
    lc_w3 = layers.fully_connected(lc_w3, 1, activation_fn=None)
    lc_w3 = tf.squeeze(lc_w3, [1])
    return lc_w0, lc_w1, lc_w2, lc_w3

@zs.reuse('discriminator1')
def logistic_discriminator1(observed, latent):
    w0 = latent['w0']
    w1 = latent['w1']
    w2 = latent['w2']
    w3 = latent['w3']

    lc_w0 = layers.flatten(w0)
    lc_w0 = layers.fully_connected(lc_w0, 1, activation_fn=None)
    lc_w0 = tf.squeeze(lc_w0)

    lc_w1 = layers.flatten(w1)
    lc_w1 = layers.fully_connected(lc_w1, 1, activation_fn=None)
    lc_w1 = tf.squeeze(lc_w1)

    lc_w2 = layers.flatten(w2)
    lc_w2 = layers.fully_connected(lc_w2, 1, activation_fn=None)
    lc_w2 = tf.squeeze(lc_w2)

    lc_w3 = layers.flatten(w3)
    lc_w3 = layers.fully_connected(lc_w3, 1, activation_fn=None)
    lc_w3 = tf.squeeze(lc_w3)
    return lc_w0, lc_w1, lc_w2, lc_w3


def run(rng):
    tf.set_random_seed(1234)


    infer_str = ['mean field', 'fully connected', 'deconv']
    infer_func = [mean_field_variational, fully_connected_variational, deconv_variational]
    disc_str = ['fully connected', 'conv', 'simple logistic regression']
    disc_func = [fully_connected_discriminator, conv_discriminator, logistic_discriminator1]
    infer_index = 2
    disc_index = 1

    print('time = {}'.format(str(datetime.now())))
    print('variational: {}'.format(infer_str[infer_index]))
    print('discriminator: {}'.format(disc_str[disc_index]))

    variational = infer_func[infer_index]
    discriminator = disc_func[disc_index]

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.vstack([y_train, y_valid])
    np.random.seed(1234)
    N, n_x = x_train.shape
    n_classes = y_train.shape[1]
    y_train = np.argmax(y_train, 1).astype('int32')
    y_test = np.argmax(y_test, 1).astype('int32')

    # Define model parameters
    n_hiddens = [150, 150, 150]

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 50
    epoches = 500
    batch_size = 100
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_batch_size = 100
    test_iters = int(np.ceil(x_test.shape[0] / float(test_batch_size)))
    test_freq = 3
    learning_rate = 0.001
    anneal_lr_freq = 1000
    anneal_lr_rate = 0.75

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    keep_rate = tf.placeholder(tf.float32, shape=[], name='keep_rate')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.int32, shape=[None])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])
    layer_sizes = [n_x] + n_hiddens + [n_classes]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    def log_like(observed):
        model, _ = bayesianNN(observed, x, n_x, layer_sizes,
                              n_particles, keep_rate)
        log_py_xw = model.local_log_prob('y')
        return tf.reduce_mean(log_py_xw, 1, keep_dims=True) * N

    ws = variational(layer_sizes, n_particles, is_training)
    latent = dict(zip(w_names, ws))

    prior_model, _ = bayesianNN(None, x, n_x, layer_sizes,
                                n_particles, keep_rate)
    prior_outputs = prior_model.outputs(w_names)
    prior_lats = dict(zip(w_names, prior_outputs))
   
    classifier1 = lambda obs, lat: discriminator(obs, lat, n_particles)
    model_loss, infer_loss, disc_loss, infer_logits, prior_logits = zs.avb(
        log_like, classifier1, {'y': y_obs}, latent, prior_lats)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    model_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
    infer_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='variational')
    disc_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')


    model_grads = [] if not len(model_var_list) else \
        optimizer.compute_gradients(
        model_loss, var_list=model_var_list)
    infer_grads = optimizer.compute_gradients(
        infer_loss, var_list=infer_var_list)
    disc_grads = optimizer.compute_gradients(
        disc_loss, var_list=disc_var_list)

    #infer_grads = [(tf.clip_by_average_norm(grad, 10.), var) for grad, var in infer_grads]
    grads = model_grads + infer_grads + disc_grads
    infer = optimizer.apply_gradients(grads)

    infer_prob = tf.nn.sigmoid(infer_logits)
    prior_prob = tf.nn.sigmoid(prior_logits)

    # prediction: rmse & log likelihood
    observed = {}
    observed.update(latent)
    _, y_logits = bayesianNN(observed, x, n_x, layer_sizes,
                               n_particles, keep_rate)
    y_pred = tf.cast(tf.argmax(tf.reduce_mean(y_logits, 0), 1), tf.int32)
    acc = tf.reduce_mean(tf.to_float(
        tf.equal(y_pred, y)))

    params = tf.trainable_variables()
    for i in params:
        print('variable name = {}, shape = {}'.format(i.name, i.get_shape()))

    # Run the inference
    #config = tf.ConfigProto(
    #    device_count={'CPU': 1},
    #    intra_op_parallelism_threads=1,
    #    inter_op_parallelism_threads=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            indices = np.random.permutation(N)
            x_train = x_train[indices, :]
            y_train = y_train[indices]
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            mls, ils, dls = [], [], []
            ips, pps = [], []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, ml, il, dl, ip, pp= sess.run(
                    [infer, model_loss, infer_loss, disc_loss, infer_prob, prior_prob],
                    feed_dict={n_particles: lb_samples,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch,
                               is_training: True, keep_rate: 0.75})
                mls.append(ml)
                ils.append(il)
                dls.append(dl)
                ips.append(ip)
                pps.append(pp)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s):'.format(epoch, time_epoch))
            print('model loss = {}, infer loss = {}, disc loss = {}'.format(
                np.mean(mls), np.mean(ils), np.mean(dls)))
            print('infer prob = {}, for prior prob = {}'.format(
                np.mean(ips), np.mean(pps)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_mls, test_ils, test_dls = [], [], []
                test_accs = []
                iter_sizes = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                            (t + 1) * test_batch_size]
                    test_y_batch = y_test[t * test_batch_size:
                                            (t + 1) * test_batch_size]
                    test_ml, test_il, test_dl, test_acc = sess.run(
                        [model_loss, infer_loss, disc_loss, acc],
                        feed_dict={n_particles: ll_samples,
                                   x: test_x_batch, y: test_y_batch,
                                   is_training: False, keep_rate: 1.})
                    test_mls.append(test_ml)
                    test_ils.append(test_il)
                    test_dls.append(test_dl)
                    test_accs.append(test_acc)
                    iter_sizes.append(test_x_batch.shape[0])

                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> TEST model loss = {}, '
                    'infer loss = {}, disc loss = {}'.format(
                    np.mean(test_mls), np.mean(test_ils), np.mean(test_dls)))
                test_acc = np.sum(np.array(test_accs) * np.array(iter_sizes)) / \
                    np.sum(iter_sizes)
                print('>> Test acc = {}'.format(test_acc))

if __name__ == '__main__':
    np.random.seed(1234)
    rng = np.random.RandomState(1)

    print(__file__)
    with open(__file__) as f:
        print(f.read())
    run(rng)
