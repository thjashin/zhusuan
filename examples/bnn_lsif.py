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
from tensorflow.contrib import layers
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

        y_mean = tf.squeeze(ly_x, [2, 3])
        y_logstd = np.log(0.25)
        y = zs.Normal('y', y_mean, y_logstd * tf.ones_like(y_mean))

    return model, y_mean


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
    return ws[0], ws[1]


@zs.reuse('variational')
def deconv_variational(layer_sizes, n_particles):
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
        return w0, w1


def run(dataset_name, logger, rng):
    tf.reset_default_graph()
    tf.set_random_seed(1234)

    @zs.reuse('variational')
    def fully_connected_variational(layer_sizes, n_particles):
        w0 = tf.random_normal([n_particles, 100])
        w0 = layers.fully_connected(w0, 1024)
        N = layer_sizes[1] * (layer_sizes[0] + 1)
        w0 = layers.fully_connected(w0, N)
        w0 = layers.fully_connected(w0, N, activation_fn=None)
        w0 = tf.reshape(w0,
                        [n_particles, 1, layer_sizes[1], layer_sizes[0] + 1])

        w1 = tf.random_normal([n_particles, 50])
        w1 = layers.fully_connected(w1, 100)
        w1 = layers.fully_connected(w1, 51)
        w1 = layers.fully_connected(w1, 51, activation_fn=None)
        w1 = tf.reshape(w1, [n_particles, 1, 1, 51])
        return w0, w1

    infer_str = ['mean field', 'fully connected', 'deconv']
    infer_func = [
        mean_field_variational,
        fully_connected_variational,
        deconv_variational
    ]
    infer_index = 1
    logger.info('the mean was at the dimension of n_particles instead of '
                'all dimensions')
    logger.info('time = {}'.format(str(datetime.now())))
    logger.info('model: no dropout, y_logstd=log(0.25)')
    logger.info('variational: {}'.format(infer_str[infer_index]))

    variational = infer_func[infer_index]

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
    test_batch_size = 1000
    test_iters = int(np.ceil(x_test.shape[0] / float(test_batch_size)))
    test_freq = 10
    learning_rate = 0.01
    anneal_lr_freq = 1000
    anneal_lr_rate = 0.75

    # LSIF parameters
    # kernel_width = 0.05
    lambda_ = 0.001
    n_basis = 100

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    qw0, qw1 = variational(layer_sizes, n_particles)
    latent = dict(zip(w_names, [qw0, qw1]))
    prior_model, _ = bayesianNN(None, x, n_x, layer_sizes, n_particles)
    pw0, pw1 = prior_model.outputs(w_names)

    # LSIF
    def rbf_kernel(w1, w2, kernel_width):
        return tf.exp(-tf.reduce_sum(tf.square(w1 - w2), [1, 2, 3]) /
                      (2 * kernel_width ** 2))

    def phi(w, w_basis, kernel_width):
        # w: [n_particles, 1, n_out, n_in + 1]
        # w_basis: [n_basis, 1, n_out, n_in + 1]
        # phi(w): [n_particles, n_basis]
        w_row = tf.expand_dims(w, 1)
        w_col = tf.expand_dims(w_basis, 0)
        return rbf_kernel(w_row, w_col, kernel_width)

    def H(w, w_basis, kernel_width):
        # phi_w: [n_particles, n_basis]
        phi_w = phi(w, w_basis, kernel_width)
        # phi_w = tf.Print(phi_w, [phi_w], summarize=100)
        phi_w_t = tf.transpose(phi_w)
        # H: [n_basis, n_basis]
        return tf.matmul(phi_w_t, phi_w) / tf.to_float(tf.shape(w)[0])
        # phi_w = phi(w, w_basis)
        # return tf.reduce_mean(
        #     tf.expand_dims(phi_w, 2) * tf.expand_dims(phi_w, 1), 0)

    def h(w, w_basis, kernel_width):
        # h: [n_basis]
        return tf.reduce_mean(phi(w, w_basis, kernel_width), 0)

    def optimal_alpha(qw_samples, pw_samples, w_basis, kernel_width):
        H_ = H(pw_samples, w_basis, kernel_width)
        h_ = h(qw_samples, w_basis, kernel_width)
        # H_ = tf.Print(H_, [H_], summarize=10000)
        alpha = tf.matmul(
            tf.matrix_inverse(H_ + lambda_ * tf.eye(tf.shape(H_)[0])),
            tf.expand_dims(h_, 1))
        # alpha: [n_basis]
        alpha = tf.squeeze(alpha, axis=1)
        # alpha = tf.Print(alpha, [alpha], message="alpha: ", summarize=20)
        # alpha = tf.maximum(alpha, 0)
        return alpha

    def heuristic_kernel_width(w_samples, w_basis):
        n_w_samples = tf.shape(w_samples)[0]
        n_w_basis = tf.shape(w_basis)[0]
        w_samples = tf.expand_dims(w_samples, 1)
        w_basis = tf.expand_dims(w_basis, 0)
        pairwise_dist = tf.sqrt(
            tf.reduce_sum(tf.square(w_samples - w_basis), [-1, -2, -3]))
        k = n_w_samples * n_w_basis // 2
        top_k_values = tf.nn.top_k(tf.reshape(pairwise_dist, [-1]), k=k).values
        kernel_width = top_k_values[-1]
        # kernel_width = tf.Print(kernel_width, [kernel_width],
        #                         message="kernel_width: ")
        return kernel_width

    def optimal_ratio(x, qw_samples, pw_samples):
        w_samples = tf.concat([qw_samples, pw_samples], axis=0)
        w_basis = qw_samples[:n_basis]
        kernel_width = heuristic_kernel_width(w_samples, w_basis)
        alpha = optimal_alpha(qw_samples, pw_samples, w_basis, kernel_width)
        # phi_x: [N, n_basis]
        phi_x = phi(x, w_basis, kernel_width)
        ratio = tf.reduce_sum(tf.expand_dims(alpha, 0) * phi_x, 1)
        ratio = tf.maximum(ratio, 1e-8)
        # ratio: [N]
        # ratio = tf.Print(ratio, [ratio], message="ratio: ", summarize=20)
        return ratio

    def log_conditional(observed):
        model, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
        log_py_xw = model.local_log_prob('y')
        return log_py_xw * N

    observed = zs.merge_dicts(latent, {'y': y_obs})
    eq_ll = log_conditional(observed)
    w0_kl_term = -tf.reduce_mean(
        tf.log(optimal_ratio(qw0, pw0, tf.stop_gradient(qw0))))
    w1_kl_term = -tf.reduce_mean(
        tf.log(optimal_ratio(qw1, pw1, tf.stop_gradient(qw1))))
    # w0_kl_term = tf.reduce_mean(
    #     tf.log(optimal_ratio(qw0, tf.stop_gradient(qw0),
    #                          tf.stop_gradient(pw0))))
    # w1_kl_term = tf.reduce_mean(
    #     tf.log(optimal_ratio(qw1, tf.stop_gradient(qw1),
    #                          tf.stop_gradient(pw1))))
    # prior_term = tf.stop_gradient(prior_term)
    ll_term = tf.reduce_mean(eq_ll)
    lower_bound = ll_term - w0_kl_term - w1_kl_term

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

    infer_grad_mean = sum([tf.reduce_mean(abs(grad))
                           for grad, _ in infer_grads]) / len(infer_grads)
    model_grad_mean = tf.constant(0.) if not len(model_grads) else \
        sum([tf.reduce_mean(abs(grad))
             for grad, _ in model_grads]) / len(model_grads)

    # prediction: rmse & log likelihood
    observed = {}
    observed.update(latent)
    observed.update({'y': y_obs})
    model, y_mean = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
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
            lbs = []
            ims, mms = [], []
            w0_kls, w1_kls = [], []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb, im, mm, w0_kl, w1_kl = sess.run(
                    [infer, lower_bound, infer_grad_mean, model_grad_mean,
                     w0_kl_term, w1_kl_term],
                    feed_dict={n_particles: lb_samples,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
                mms.append(mm)
                ims.append(im)
                w0_kls.append(w0_kl)
                w1_kls.append(w1_kl)
            time_epoch += time.time()
            logger.info('Epoch {} ({:.1f}s):'.format(epoch, time_epoch))
            logger.info('lower bound = {}'.format(np.mean(lbs)))
            logger.info('w0_kl = {}, w1_kl = {}'.format(
                np.mean(w0_kls), np.mean(w1_kls)))
            logger.info('model_grad = {}, infer grad = {}'
                        .format(np.mean(mms), np.mean(ims)))

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

    logger = logging.getLogger('bnn_lsif')
    logger.setLevel(logging.DEBUG)
    log_path = 'logs/bnn_lsif/' + FLAGS.dataset + '/log' + \
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
