#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
import logging
from datetime import datetime

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs

from examples.utils import makedirs
import avb_dataset as dataset

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dataset", "boston", """Which dataset to use""")


@zs.reuse('model')
def bayesianNN(observed, x, n_x, layer_sizes, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([n_particles, 1, n_out, n_in + 1])
            w_logstd = tf.zeros([n_particles, 1, n_out, n_in + 1])
            ws.append(zs.Normal('w' + str(i), w_mu, w_logstd,
                                group_event_ndims=2))

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

        y_mean = tf.squeeze(ly_x, [2, 3])
        y_prec_alpha = 6. * tf.ones([n_particles, 1])
        y_prec_beta = 6. * tf.ones([n_particles, 1])
        y_prec = zs.Gamma('y_prec', y_prec_alpha, y_prec_beta)
        y_logstd = -0.5*tf.log(y_prec)

        y = zs.Normal('y', y_mean, y_logstd * tf.ones_like(y_mean))

    return model, y_mean


def run(dataset_name, logger, rng):
    tf.reset_default_graph()
    tf.set_random_seed(1234)

    @zs.reuse('variational')
    def mean_field_variational(layer_sizes, n_particles):
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
        return variational

    infer_str = ['mean field', 'fully connected', 'deconv']
    infer_func = [
        mean_field_variational,
    ]
    infer_index = 0
    logger.info('the mean was at the dimension of n_particles instead of '
                'all dimensions')
    logger.info('time = {}'.format(str(datetime.now())))
    logger.info('model: no dropout, y_logstd=gamma')
    logger.info('variational: {}'.format(infer_str[infer_index]))

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
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    print(x_train.shape, x_test.shape)
    y_train, y_test, mean_y_train, std_y_train = \
        dataset.standardize(y_train, y_test)

    # Define model parameters
    if (dataset == "year") or (dataset == "protein"):
        n_hiddens = [100]
    else:
        n_hiddens = [50]

    # Define training/evaluation parameters
    lb_samples = 100
    ll_samples = 1000
    epoches = 300
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_batch_size = 100
    test_iters = int(np.ceil(x_test.shape[0] / float(test_batch_size)))
    test_freq = 10
    learning_rate = 0.01
    anneal_lr_freq = 1000
    anneal_lr_rate = 0.75

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    with tf.variable_scope('variational'):
        with zs.BayesianNet() as variational_prec:
            prec_logalpha = tf.get_variable(
                'prec_logalpha', shape=[1],
                initializer=tf.constant_initializer(np.log(6.)))
            prec_logbeta = tf.get_variable(
                'prec_logbeta', shape=[1],
                initializer=tf.constant_initializer(np.log(6.)))
            alpha = tf.exp(prec_logalpha)
            beta = tf.exp(prec_logbeta)
            q_prec = zs.Gamma('y_prec', alpha, beta, n_samples=n_particles,
                              group_event_ndims=1)
            q_prec = tf.stop_gradient(q_prec)
            kldiv_prec = tf.contrib.distributions.kl(
                tf.contrib.distributions.Gamma(alpha, beta),
                tf.contrib.distributions.Gamma(6., 6.))

    def log_joint(observed):
        model, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
        log_pws = model.local_log_prob(w_names)
        log_py_xw = model.local_log_prob('y')
        return tf.add_n(log_pws) + log_py_xw * N

    variational = mean_field_variational(layer_sizes, n_particles)
    qw_outputs = variational.query(w_names, outputs=True,
                                   local_log_prob=True)
    # qw0_outputs, qw1_outputs = variational.query(w_names, outputs=True,
    #                                              local_log_prob=True)
    # qw0_samples, log_qw0 = qw0_outputs
    # qw1_samples, log_qw1 = qw1_outputs
    # qw0_samples = tf.reshape(qw0_samples, [n_particles, 1, n_hiddens[0] * (n_x + 1)])
    # qw1_samples = tf.reshape(qw1_samples, [n_particles, 1, 1 * (n_hiddens[0] + 1)])
    # qw0_samples, log_qw0 = zs.planar_normalizing_flow(qw0_samples, log_qw0,
    #                                                   n_iters=10)
    # qw1_samples, log_qw1 = zs.planar_normalizing_flow(qw1_samples, log_qw1,
    #                                                   n_iters=10)
    # qw0_samples = tf.reshape(qw0_samples, [n_particles, 1, 50, n_x + 1])
    # qw1_samples = tf.reshape(qw1_samples, [n_particles, 1, 1, 50 + 1])
    # qw_outputs = [(qw0_samples, log_qw0), (qw1_samples, log_qw1)]
    latent = dict(zip(w_names, qw_outputs))
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'y': y_obs, 'y_prec': q_prec}, latent, axis=0))

    observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
    observed.update({'y': y_obs})
    observed.update({'y_prec': q_prec}) #NOTE
    model, y_mean = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
    #loss for precision
    loss_prec = 0.5 * N *\
            (tf.stop_gradient(tf.reduce_mean((y_obs-y_mean)**2))\
            * alpha / beta - \
            (tf.digamma(alpha) - tf.log(beta+1e-10))) + kldiv_prec
    lower_bound = lower_bound - loss_prec

    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    model_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
    infer_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='variational')

    model_grads = [] if not len(model_var_list) else \
        optimizer.compute_gradients(
            -lower_bound, var_list=model_var_list)
    infer_grads = optimizer.compute_gradients(
        -lower_bound, var_list=infer_var_list)

    infer_grads = [(tf.clip_by_average_norm(grad, 10.), var)
                   for grad, var in infer_grads]
    grads = model_grads + infer_grads
    infer = optimizer.apply_gradients(grads)

    # prediction: rmse & log likelihood
    y_pred = tf.reduce_mean(y_mean, 0)
    rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
    log_py_xw = model.local_log_prob('y')
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
        tf.log(std_y_train)

    params = tf.trainable_variables()
    for i in params:
        logger.info('variable name = {}, shape = {}'
                    .format(i.name, i.get_shape()))

    # Run the inference
    test_rmse_result = []
    test_ll_result = []
    # config = tf.ConfigProto(
    #     device_count={'CPU': 1},
    #     intra_op_parallelism_threads=1,
    #     inter_op_parallelism_threads=1)
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            time_epoch = -time.time()
            indices = np.random.permutation(N)
            x_train = x_train[indices, :]
            y_train = y_train[indices]
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run(
                    [infer, lower_bound],
                    feed_dict={n_particles: lb_samples,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
            time_epoch += time.time()
            logger.info('Epoch {} ({:.1f}s):'.format(epoch, time_epoch))
            logger.info('lower bound = {}'.format(np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs, test_rmses, test_lls = [], [], []
                iter_sizes = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_y_batch = y_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(
                        lower_bound,
                        feed_dict={n_particles: lb_samples,
                                   x: test_x_batch, y: test_y_batch})
                    test_rmse, test_ll = sess.run(
                        [rmse, log_likelihood],
                        feed_dict={n_particles: ll_samples,
                                   x: test_x_batch, y: test_y_batch})
                    test_lbs.append(test_lb)
                    test_rmses.append(test_rmse)
                    iter_sizes.append(test_x_batch.shape[0])
                    test_lls.append(test_ll)
                time_test += time.time()
                logger.info('>>> TEST ({:.1f}s)'.format(time_test))
                logger.info(
                    '>> TEST lower bound = {}'.format(np.mean(test_lbs)))
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

    logger = logging.getLogger('bnn_flow_gamma')
    logger.setLevel(logging.DEBUG)
    log_path = 'logs/bnn_flow_gamma/' + FLAGS.dataset + '/log' + \
        time.strftime("%Y%m%d-%H%M%S")
    makedirs(log_path)
    info_file_handler = logging.FileHandler(log_path)
    info_file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(info_file_handler)
    logger.addHandler(console_handler)
    logger.info(__file__)
    with open(__file__) as f:
        logger.info(f.read())

    n_runs = 20
    if FLAGS.dataset == "protein":
        n_runs = 5
    elif FLAGS.dataset == "year":
        n_runs = 1
    print("Dataset = {}, N_RUNS = {}".format(FLAGS.dataset, n_runs))
    rmse_results = []
    ll_results = []
    for _ in range(1, n_runs + 1):
        rmse_result, ll_result = run(FLAGS.dataset, logger, rng)
        rmse_results.append(rmse_result)
        ll_results.append(ll_result)

    for i, (rmse_result, ll_result) in enumerate(zip(rmse_results,
                                                     ll_results)):
        logger.info("\n## RUN {}".format(i))
        logger.info('# Test rmse = {}'.format(rmse_result))
        logger.info('# Test log likelihood = {}'.format(ll_result))

    for i in range(len(rmse_results[0])):
        logger.info("\n## AVERAGE for {}".format(i))
        test_rmses = [a[i] for a in rmse_results]
        test_lls = [a[i] for a in ll_results]

        logger.info("Test rmse = {}/{}".format(
            np.mean(test_rmses), np.std(test_rmses) / n_runs ** 0.5))
        logger.info("Test log likelihood = {}/{}".format(
            np.mean(test_lls), np.std(test_lls) / n_runs ** 0.5))
        logger.info('NOTE: Test result above output mean and std. errors')
