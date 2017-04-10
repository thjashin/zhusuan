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
def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def avb(log_like, classifier1, classifier2, observed, latent, prior):
    #sample mean distribution of latent and prior
    l_w0, l_w1 = latent['w0'], latent['w1']
    p_w0, p_w1 = prior['w0'], prior['w1']
    z0 = tf.random_uniform(tf.shape(l_w0)[:1])
    z0 = tf.to_float(tf.greater(z0, 0.5))
    z1 = tf.random_uniform(tf.shape(l_w1)[:1])
    z1 = tf.to_float(tf.greater(z1, 0.5))
    def _mul_1st_dim(z, w):
        return tf.transpose(z * tf.transpose(w))
    mean_w0 = _mul_1st_dim(z0, l_w0) + _mul_1st_dim(1-z0, p_w0)
    mean_w1 = _mul_1st_dim(z1, l_w1) + _mul_1st_dim(1-z1, p_w1)
   # mean_w0 = z0 * l_w0 + (1-z0) * p_w0
   # mean_w1 = z1 * l_w1 + (1-z1) * p_w1
    mean = {'w0':mean_w0, 'w1':mean_w1}

    infer_class_logits1 = classifier1(observed, latent)
    mean_class_logits1 = classifier1(observed, mean)
    mean_class_logits2 = classifier2(observed, mean)
    prior_class_logits2 = classifier2(observed, prior)
    infer_class_logits2 = classifier2(observed, latent)

    if not type(infer_class_logits1) == type((0,)):
        infer_class_logits1 = [infer_class_logits1]
        prior_class_logits2 = [prior_class_logits2]
        mean_class_logits1 = [mean_class_logits1]
        mean_class_logits2 = [mean_class_logits2]
    infer_class_logits1 = list(infer_class_logits1)
    prior_class_logits2 = list(prior_class_logits2)
    mean_class_logits1 = list(mean_class_logits1)
    mean_class_logits2 = list(mean_class_logits2)

    joint_obs = merge_dicts(observed, latent)
    model_loss = -tf.reduce_mean(log_like(joint_obs))
    infer_loss = model_loss +\
        sum([tf.reduce_mean(il) for il in infer_class_logits1]) +\
        sum([tf.reduce_mean(il) for il in infer_class_logits2])  
#        sum([tf.reduce_mean(ml) for ml in mean_class_logits2])
    disc_loss1 = sum([
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(il), logits=il)) + \
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(ml), logits=ml))
        for il, ml in zip(infer_class_logits1, mean_class_logits1)])
    disc_loss2 = sum([
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(ml), logits=ml)) + \
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(pl), logits=pl))
        for ml, pl in zip(mean_class_logits2, prior_class_logits2)])
    disc_loss = disc_loss1 + disc_loss2
    return model_loss, infer_loss, disc_loss, infer_class_logits1, prior_class_logits2



@zs.reuse('model')
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
#                ly_x = layers.dropout(ly_x, keep_rate)

        y_mean = tf.squeeze(ly_x, [2, 3])
        
#        y_logstd_mean = tf.get_variable('y_logstd_mean', shape=[],
#                                        initializer=tf.constant_initializer(-1.5))
#        y_logstd_logstd = tf.get_variable('y_logstd_logstd', shape=[],
#                                        initializer=tf.constant_initializer(0.5))
#        y_logstd = zs.Normal('y_logstd', y_logstd_mean, y_logstd_logstd)
        y_logstd = np.log(0.25)
        #y_logstd = tf.get_variable('y_logstd', shape=[],
        #                           initializer=tf.constant_initializer(0.))
        y = zs.Normal('y', y_mean, y_logstd * tf.ones_like(y_mean))

    return model, y_mean

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

    w0 = tf.random_normal([n_particles, 100])
    w0 = layers.fully_connected(w0, 24)
    w0 = tf.reshape(w0, [n_particles, 8, 3, 1])
    w0 = layers.conv2d_transpose(w0, 4, 4, stride=[2, 3])
    w0 = layers.conv2d_transpose(w0, 4, 4, stride=[3, 1])
    w0 = layers.conv2d_transpose(w0, 1, [3, 1], stride=1,
                                  padding='VALID', activation_fn=None)
    w0 = tf.transpose(w0, [0, 3, 1, 2])
    _assert_w0_shape = tf.assert_equal(tf.shape(w0),
                                       [n_particles, 1, 50, 9], 
                                       message="w0 shape wrong")

    w1 = tf.random_normal([n_particles, 50])
    w1 = layers.fully_connected(w1, 12)
    w1 = tf.reshape(w1, [n_particles, 1, 12, 1])
    w1 = layers.conv2d_transpose(w1, 4, 4, stride=[1, 2])
    w1 = layers.conv2d_transpose(w1, 4, 4, stride=[1, 2])
    w1 = layers.conv2d_transpose(w1, 1, [1, 4], stride=1,
                                 padding='VALID', activation_fn=None)
    w1 = tf.transpose(w1, [0, 3, 1, 2])
    _assert_w1_shape = tf.assert_equal(tf.shape(w1),
                                       [n_particles, 1, 1, 51],
                                       message='w1 shape wrong')
    with tf.control_dependencies([_assert_w0_shape, _assert_w1_shape]):
        return  w0, w1


@zs.reuse('discriminator')
def fully_connected_discriminator(observed, latent):
    w0 = latent['w0']
    w1 = latent['w1']

    lc_w0 = layers.flatten(w0)
    lc_w0 = layers.fully_connected(lc_w0, 1024)
    lc_w0 = layers.fully_connected(lc_w0, 1024)
    lc_w0 = layers.fully_connected(lc_w0, 1, activation_fn=None)
    lc_w0 = tf.squeeze(lc_w0)

    lc_w1 = layers.flatten(w1)
    lc_w1 = layers.fully_connected(lc_w1, 100)
    lc_w1 = layers.fully_connected(lc_w1, 100)
    lc_w1 = layers.fully_connected(lc_w1, 1, activation_fn=None)
    lc_w1 = tf.squeeze(lc_w1)

    return lc_w0, lc_w1


@zs.reuse('discriminator')
def conv_discriminator(observed, latent):
    w0 = tf.transpose(latent['w0'], [0, 2, 3, 1])
    w1 = tf.transpose(latent['w1'], [0, 2, 3, 1])

    nwf = 8
    lc_w0 = layers.conv2d(w0, nwf, 5, stride=2)
    lc_w0 = layers.conv2d(lc_w0, nwf*2, 5, stride=2)
    lc_w0 = layers.flatten(lc_w0)
    lc_w0 = layers.fully_connected(lc_w0, 1, activation_fn=None)
    lc_w0 = tf.squeeze(lc_w0)

    lc_w1 = layers.conv2d(w1, nwf, 5, stride=2)
    lc_w1 = layers.conv2d(lc_w1, nwf*2, 5, stride=2)
    lc_w1 = layers.flatten(lc_w1)
    lc_w1 = layers.fully_connected(lc_w1, 1, activation_fn=None)
    lc_w1 = tf.squeeze(lc_w1)
    return lc_w0, lc_w1

@zs.reuse('discriminator1')
def logistic_discriminator1(observed, latent):
    w0 = tf.transpose(latent['w0'], [0, 2, 3, 1])
    w1 = tf.transpose(latent['w1'], [0, 2, 3, 1])

    lc_w0 = layers.flatten(w0)
    lc_w0 = layers.fully_connected(lc_w0, 1, activation_fn=None)
    lc_w0 = tf.squeeze(lc_w0)

    lc_w1 = layers.flatten(w1)
    lc_w1 = layers.fully_connected(lc_w1, 1, activation_fn=None)
    lc_w1 = tf.squeeze(lc_w1)
    return lc_w0, lc_w1

@zs.reuse('discriminator2')
def logistic_discriminator2(observed, latent):
    w0 = tf.transpose(latent['w0'], [0, 2, 3, 1])
    w1 = tf.transpose(latent['w1'], [0, 2, 3, 1])

    lc_w0 = layers.flatten(w0)
    lc_w0 = layers.fully_connected(lc_w0, 1, activation_fn=None)
    lc_w0 = tf.squeeze(lc_w0)

    lc_w1 = layers.flatten(w1)
    lc_w1 = layers.fully_connected(lc_w1, 1, activation_fn=None)
    lc_w1 = tf.squeeze(lc_w1)
    return lc_w0, lc_w1


def run(dataset_name, logger, rng):
    tf.reset_default_graph()
    tf.set_random_seed(1234)


    infer_str = ['mean field', 'fully connected', 'deconv']
    infer_func = [mean_field_variational, fully_connected_variational, deconv_variational]
    disc_str = ['fully connected', 'conv', 'simple logistic regression']
    disc_func = [fully_connected_discriminator, conv_discriminator, logistic_discriminator1]
    infer_index = 1
    disc_index = 2
    logger.info("split T to two terms")
    logger.info('the mean was at the dimension of n_particles instead of all dimensions')

    logger.info('time = {}'.format(str(datetime.now())))
    logger.info('model: no dropout, y_logstd=log(0.25)')
    logger.info('variational: {}'.format(infer_str[infer_index]))
    logger.info('discriminator: {}'.format(disc_str[disc_index]))

    variational = infer_func[infer_index]
    discriminator = disc_func[disc_index]

    # Load UCI Boston housing data
    datasets = dict(
        boston='housing.data',
        concrete='concrete.data',
        energy='energy.data',
        kin8nm='kin8nm.data',
        naval='naval.data',
        power_plant='power_plant.data',
        protein='protein.data',
        wine='wine.data',
        yacht='yacht_hydrodynamics.data',
        year='year_prediction.data',
    )
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', datasets[dataset_name])
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_uci_datasets(data_path, rng, delimiter=None)

    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    logger.info('dataset name = {}'.format(dataset_name))
    logger.info('x_train shape = {}'.format(x_train.shape))
    logger.info('x_test shape = {}'.format(x_test.shape))
    N, n_x = x_train.shape

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train,
                                                x_test)
    y_train, y_test, mean_y_train, std_y_train = \
        dataset.standardize(
            y_train.reshape((-1, 1)),
            y_test.reshape((-1, 1)))
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    std_y_train = std_y_train.squeeze()

    # Define model parameters
    n_hiddens = [50]

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 1000
    epoches = 300
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_batch_size = 1000
    test_iters = int(np.ceil(x_test.shape[0] / float(test_batch_size)))
    test_freq = 10
    learning_rate = 0.01
    anneal_lr_freq = 1000
    anneal_lr_rate = 0.75

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    keep_rate = tf.placeholder(tf.float32, shape=[], name='keep_rate')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    def log_like(observed):
        model, _ = bayesianNN(observed, x, n_x, layer_sizes,
                              n_particles, keep_rate)
        log_py_xw = model.local_log_prob('y')
        return tf.reduce_mean(log_py_xw, 1, keep_dims=True) * N

    w0, w1 = variational(layer_sizes, n_particles, is_training)
    latent = dict(zip(w_names, [w0, w1]))

    prior_model, _ = bayesianNN(None, x, n_x, layer_sizes,
                                n_particles, keep_rate)
    prior_outputs = prior_model.outputs(w_names)
    prior_lats = dict(zip(w_names, prior_outputs))

    classifier1 = lambda obs, lat: logistic_discriminator1(obs, lat)
    classifier2 = lambda obs, lat: logistic_discriminator2(obs, lat)
    model_loss, infer_loss, disc_loss, infer_logits, prior_logits = avb(
        log_like, classifier1, classifier2, {'y': y_obs}, latent, prior_lats)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    model_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
    infer_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='variational')
    disc_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator1') +\
        tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator2')


    model_grads = [] if not len(model_var_list) else \
        optimizer.compute_gradients(
        model_loss, var_list=model_var_list)
    infer_grads = optimizer.compute_gradients(
        infer_loss, var_list=infer_var_list)
    disc_grads = optimizer.compute_gradients(
        disc_loss, var_list=disc_var_list)

    infer_grads = [(tf.clip_by_average_norm(grad, 10.), var) for grad, var in infer_grads]
    grads = model_grads + infer_grads + disc_grads
    infer = optimizer.apply_gradients(grads)

    infer_prob = tf.nn.sigmoid(infer_logits)
    prior_prob = tf.nn.sigmoid(prior_logits)
    infer_grad_mean = sum([tf.reduce_mean(abs(grad)) for grad, _ in infer_grads]) / len(infer_grads)
    model_grad_mean = tf.constant(0.) if not len(model_grads) else \
        sum([tf.reduce_mean(abs(grad)) for grad, _ in model_grads]) / len(model_grads)
    disc_grad_mean = sum([tf.reduce_mean(abs(grad)) for grad, _ in disc_grads]) / len(disc_grads)

    # prediction: rmse & log likelihood
    observed = {}
    observed.update(latent)
    observed.update({'y': y_obs})
    model, y_mean = bayesianNN(observed, x, n_x, layer_sizes,
                               n_particles, keep_rate)
    y_pred = tf.reduce_mean(y_mean, 0)
    rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
    log_py_xw = model.local_log_prob('y')
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
        tf.log(std_y_train)

    params = tf.trainable_variables()
    for i in params:
        logger.info('variable name = {}, shape = {}'.format(i.name, i.get_shape()))

    # Run the inference
    test_rmse_result = []
    test_ll_result = []
    config = tf.ConfigProto(
        device_count={'CPU': 1},
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    with tf.Session(config=config) as sess:
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
            ims, mms, dms = [], [], []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, ml, il, dl, ip, pp, im, mm, dm= sess.run(
                    [infer, model_loss, infer_loss, disc_loss, infer_prob, prior_prob, infer_grad_mean, model_grad_mean, disc_grad_mean],
                    feed_dict={n_particles: lb_samples,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch,
                               is_training: True, keep_rate: 0.75})
                mls.append(ml)
                ils.append(il)
                dls.append(dl)
                mms.append(mm)
                ims.append(im)
                dms.append(dm)
                ips.append(ip)
                pps.append(pp)
            time_epoch += time.time()
            logger.info('Epoch {} ({:.1f}s):'.format(epoch, time_epoch))
            logger.info('model loss = {}, infer loss = {}, disc loss = {}'.format(
                np.mean(mls), np.mean(ils), np.mean(dls)))
            logger.info('infer prob = {}, for prior prob = {}'.format(
                np.mean(ips), np.mean(pps)))
            logger.info('model_grad = {}, infer grad = {}, disc_grad = {}'.format(
                np.mean(mms), np.mean(ims), np.mean(dms)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_mls, test_ils, test_dls = [], [], []
                test_rmses, test_lls = [], []
                iter_sizes = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                            (t + 1) * test_batch_size]
                    test_y_batch = y_test[t * test_batch_size:
                                            (t + 1) * test_batch_size]
                    test_ml, test_il, test_dl, test_rmse, test_ll = sess.run(
                        [model_loss, infer_loss, disc_loss,
                         rmse, log_likelihood],
                        feed_dict={n_particles: ll_samples,
                                   x: test_x_batch, y: test_y_batch,
                                   is_training: False, keep_rate: 1.})
                    test_mls.append(test_ml)
                    test_ils.append(test_il)
                    test_dls.append(test_dl)
                    test_rmses.append(test_rmse)
                    iter_sizes.append(test_x_batch.shape[0])
                    test_lls.append(test_ll)
                time_test += time.time()
                logger.info('>>> TEST ({:.1f}s)'.format(time_test))
                logger.info('>> TEST model loss = {}, '
                    'infer loss = {}, disc loss = {}'.format(
                    np.mean(test_mls), np.mean(test_ils), np.mean(test_dls)))
                test_rmse_ = np.sqrt(
                    np.sum(np.array(test_rmses) ** 2 * np.array(iter_sizes)) /
                    np.sum(iter_sizes))
                test_ll_ = np.mean(test_lls)
                logger.info('>> Test rmse = {}'.format(test_rmse_))
                logger.info('>> Test log likelihood = {}'.format(test_ll_))
                test_rmse_result.append(test_rmse_)
                test_ll_result.append(test_ll_)
                logger.info('>> ALL Test rmse = {}'.format(test_rmse_result))
                logger.info('>> ALL Test ll = {}'.format(test_ll_result))


        # Final results
        return test_rmse_result, test_ll_result


if __name__ == '__main__':
    np.random.seed(1234)
    rng = np.random.RandomState(1)

    dataset_name = 'wine'
    logger = logging.getLogger('avb_bnn')
    logger.setLevel(logging.DEBUG)
    info_file_handler = logging.FileHandler('logs/avb_bnn_split/'+dataset_name+'.log')
    info_file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(info_file_handler)
    logger.addHandler(console_handler)
    logger.info(__file__)
    with open(__file__) as f:
        logger.info(f.read())

    iter_run = 20
    rmse_results = []
    ll_results = []
    for _ in range(1, iter_run + 1):
        rmse_result, ll_result = run(dataset_name, logger, rng)
        rmse_results.append(rmse_result)
        ll_results.append(ll_result)

    for i, (rmse_result, ll_result) in enumerate(zip(rmse_results, ll_results)):
        logger.info("\n## RUN {}".format(i))
        logger.info('# Test rmse = {}'.format(rmse_result))
        logger.info('# Test log likelihood = {}'.format(ll_result))

    for i in range(len(rmse_results[0])):
         logger.info("\n## AVERAGE for {}".format(i))
         test_rmses = [a[i] for a in rmse_results]
         test_lls = [a[i] for a in ll_results]

         logger.info("Test rmse = {}/{}".format(np.mean(test_rmses), np.std(test_rmses) / iter_run**0.5))
         logger.info("Test log likelihood = {}/{}".format(np.mean(test_lls),
                                               np.std(test_lls) / iter_run **0.5))
         logger.info('NOTE: Test result above output mean and std. errors')

