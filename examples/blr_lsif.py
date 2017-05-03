#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import zhusuan as zs

# For Mac OS
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import cm, colors


@zs.reuse('model')
def blr(observed, D, x, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        # w: [n_particles, D]
        w = zs.Normal('w', tf.zeros([D]), tf.ones([D]), n_samples=n_particles,
                      group_event_ndims=1)
        # x: [N, D]
        pred = tf.matmul(w, tf.transpose(x))
        # y: [n_particles, N]
        y = zs.Bernoulli('y', pred, dtype=tf.float32)
    return model


@zs.reuse('variational')
def q_net(observed):
    with zs.BayesianNet(observed=observed) as variational:
        w_mean = tf.get_variable('w_mean', shape=[D], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.))
        w_logstd = tf.get_variable('w_logstd', shape=[D], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.))
        # w: [n_particles, D]
        w = zs.Normal('w', w_mean, w_logstd, n_samples=n_particles,
                      group_event_ndims=1)
    return variational


def kde(samples, points, kde_stdev):
    # samples: [n D]
    # points: [m D]
    # return [m]
    samples = tf.expand_dims(samples, 1)
    points = tf.expand_dims(points, 0)
    # [n m D]
    Z = np.sqrt(2 * np.pi) * kde_stdev
    log_probs = -np.log(Z) + (-0.5 / kde_stdev ** 2) * (
        samples - points) ** 2
    log_probs = tf.reduce_sum(log_probs, -1)
    log_probs = zs.log_mean_exp(log_probs, 0)

    return log_probs


def compute_tickers(probs, n_bins=20):
    # Sort
    flat_probs = list(probs.flatten())
    flat_probs.sort()
    flat_probs.reverse()

    num_intervals = n_bins * (n_bins - 1) // 2
    interval_size = len(flat_probs) // num_intervals

    tickers = []
    cnt = 0
    for i in range(n_bins - 1):
        tickers.append(flat_probs[cnt])
        cnt += interval_size * (i + 1)
    tickers.append(flat_probs[-1])
    tickers.reverse()
    return tickers


def contourf(x, y, z):
    tickers = compute_tickers(z)
    palette = cm.PuBu
    plt.contourf(x, y, z, tickers,
                 cm=palette,
                 norm=colors.BoundaryNorm(tickers, ncolors=palette.N))
    plt.colorbar()


if __name__ == "__main__":
    tf.set_random_seed(1235)
    np.random.seed(1234)

    # Define model parameters
    N = 200
    D = 2
    learning_rate = 1
    learning_rate_g = 0.01
    learning_rate_d = 0.003
    t0 = 100
    t0_d = 100
    t0_g = 100
    epoches = 100
    epoches_d = 10
    epoches_d0 = 1000
    epoches_g = 500
    gen_n_samples = 100
    lower_box = -5
    upper_box = 5
    kde_batch_size = 2000
    n_qw_samples = 10000
    kde_stdev = 0.05
    plot_interval = 200

    # LSIF parameters
    # kernel_width = 0.5
    lambda_ = 0.01

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[N, D], name='x')
    y = tf.placeholder(tf.float32, shape=[N], name='y')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    # y_rep = tf.tile(tf.expand_dims(y, axis=1), [1, n_particles])
    # [n_particles, N]
    y_obs = tf.tile(tf.expand_dims(y, axis=0), [n_particles, 1])

    # Generate synthetic data
    model = blr({}, D, x, n_particles)
    pw_outputs, py_outputs = model.query(['w', 'y'], outputs=True,
                                         local_log_prob=True)
    pw_samples, log_pw = pw_outputs
    py_samples = py_outputs[0]

    # Variational inference
    def log_joint(observed):
        model = blr(observed, D, x, n_particles)
        # log_pw: [n_particles]; log_py_w: [n_particles, N]
        log_pw, log_py_w = model.local_log_prob(['w', 'y'])
        # [n_particles]
        return log_pw + tf.reduce_sum(log_py_w, 1), log_pw, \
            tf.reduce_sum(log_py_w, 1)

    # MFVI
    variational = q_net({})
    # [n_particles, D], [n_particles]
    vw_samples, log_qw = variational.query('w', outputs=True,
                                           local_log_prob=True)
    lower_bound = tf.reduce_mean(
        log_joint({'w': vw_samples, 'y': y_obs})[0] - log_qw)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    v_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope="variational")
    infer = optimizer.minimize(-lower_bound, var_list=v_parameters)

    # LSIF
    # Generator
    with tf.name_scope('generator'):
        epsilon = tf.random_normal([n_particles, D])
        h = layers.fully_connected(epsilon, 20, scope="generator1")
        h = layers.fully_connected(h, 20, scope="generator2")
        h = layers.fully_connected(h, 20, scope="generator3")
        # [n_particles, D]
        qw_samples = layers.fully_connected(h, D, activation_fn=None,
                                            scope="generator4")

    def rbf_kernel(w1, w2, kernel_width):
        return tf.exp(-tf.reduce_sum(tf.square(w1 - w2), -1) /
                      (2 * kernel_width ** 2))

    def phi(w, w_basis, kernel_width):
        # w: [n_particles, D]
        # w_basis: [n_basis, D]
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

    def optimal_alpha(qw_samples, pw_samples, kernel_width):
        H_ = H(pw_samples, qw_samples, kernel_width)
        h_ = h(qw_samples, qw_samples, kernel_width)
        # H_ = tf.Print(H_, [H_], summarize=10000)
        alpha = tf.matmul(
            tf.matrix_inverse(H_ + lambda_ * tf.eye(tf.shape(H_)[0])),
            tf.expand_dims(h_, 1))
        # alpha: [n_basis] = [n_particles]
        alpha = tf.squeeze(alpha, axis=1)
        alpha = tf.Print(alpha, [alpha], message="alpha: ", summarize=20)
        # alpha = tf.maximum(alpha, 0)
        return alpha

    def heuristic_kernel_width(w_samples, w_basis):
        n_w_samples = tf.shape(w_samples)[0]
        n_w_basis = tf.shape(w_basis)[0]
        w_samples = tf.expand_dims(w_samples, 1)
        w_basis = tf.expand_dims(w_basis, 0)
        pairwise_dist = tf.sqrt(
            tf.reduce_sum(tf.square(w_samples - w_basis), -1))
        k = n_w_samples * n_w_basis // 2
        top_k_values = tf.nn.top_k(tf.reshape(pairwise_dist, [-1]), k=k).values
        kernel_width = top_k_values[-1]
        kernel_width = tf.Print(kernel_width, [kernel_width],
                                message="kernel_width: ")
        return kernel_width

    def optimal_ratio(x, qw_samples, pw_samples):
        w_samples = tf.concat([qw_samples, pw_samples], axis=0)
        w_basis = qw_samples
        kernel_width = heuristic_kernel_width(w_samples, w_basis)
        alpha = optimal_alpha(qw_samples, pw_samples, kernel_width)
        # phi_x: [N, n_basis]
        phi_x = phi(x, qw_samples, kernel_width)
        ratio = tf.reduce_sum(tf.expand_dims(alpha, 0) * phi_x, 1)
        ratio = tf.maximum(ratio, 1e-8)
        # ratio: [N]
        return tf.Print(ratio, [ratio], message="ratio: ", summarize=20)

    eq_ll = log_joint({'w': qw_samples, 'y': y_obs})[2]
    prior_term = tf.reduce_mean(
        tf.log(optimal_ratio(qw_samples, tf.stop_gradient(qw_samples),
                             tf.stop_gradient(pw_samples))))
    # prior_term = tf.stop_gradient(prior_term)
    ll_term = tf.reduce_mean(-eq_ll)
    gen_obj = prior_term + ll_term

    g_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope="generator")
    g_optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    g_infer = g_optimizer.minimize(gen_obj, var_list=g_parameters)

    def check_grads(loss, var_list):
        grads_to_be_check = optimizer.compute_gradients(loss,
                                                        var_list=var_list)
        if not grads_to_be_check:
            raise ValueError("No gradients for the given var_list.")
        grads_to_be_check_ = [g for g, v in grads_to_be_check if g is not None]
        grads_to_be_check_mean = tf.reduce_mean(
            tf.stack(
                [tf.reduce_mean(tf.square(g)) for g in grads_to_be_check_]))
        grads_to_be_check_var = tf.reduce_mean(
            tf.stack([tf.nn.moments(g, list(range(0, g.get_shape().ndims)))[1]
                      for g in grads_to_be_check_]))
        return grads_to_be_check_mean, grads_to_be_check_var

    ll_grad_mean, ll_grad_var = check_grads(ll_term, g_parameters)
    prior_grad_mean, prior_grad_var = check_grads(prior_term, g_parameters)

    # Plotting
    w_ph = tf.placeholder(tf.float32, shape=[None, D], name='w_ph')
    log_joint_value, log_prior, _ = log_joint({'w': w_ph, 'y': y_obs})
    log_mean_field = q_net({'w': w_ph}).local_log_prob('w')
    # KDE
    samples_ph = tf.placeholder(tf.float32, shape=[None, D], name='samples')
    points_ph = tf.placeholder(tf.float32, shape=[None, D], name='points')
    grid_prob_op = kde(samples_ph, points_ph, kde_stdev)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Generate data
        train_x = np.random.rand(N, D) * (upper_box - lower_box) + lower_box
        train_w_sample, train_y = sess.run(
            [pw_samples, py_samples], feed_dict={x: train_x, n_particles: 1})
        # train_w_sample: [D]
        train_w_sample = np.squeeze(train_w_sample)
        print("decision boundary: {}x1 + {}x2 = 0".format(train_w_sample[0],
                                                          train_w_sample[1]))
        # train_y: [N]
        train_y = np.squeeze(train_y)

        # Run the mean-field variational inference
        for epoch in range(1, epoches + 1):
            lr = learning_rate * t0 / (t0 + epoch - 1)
            time_epoch = -time.time()
            _, lb = sess.run([infer, lower_bound],
                             feed_dict={x: train_x,
                                        y: train_y,
                                        learning_rate_ph: lr,
                                        n_particles: 100})
            time_epoch += time.time()
            if epoch % 100 == 0:
                print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                    epoch, time_epoch, lb))

        # Run the LSIF-based inference
        print('Starting LSIF-based training...')
        gen_objs = []
        for epoch in range(1, epoches_g + 1):
            lr = learning_rate_g * t0_g / (t0_g + epoch - 1)

            _, go, pv, lv = sess.run([g_infer, gen_obj, prior_term, ll_term],
                                     feed_dict={x: train_x,
                                                y: train_y,
                                                learning_rate_ph: lr,
                                                n_particles: gen_n_samples})
            gen_objs.append(go)
            print('Epoch {}, Generator obj = {}, prior = {}, ll = {}'
                  .format(epoch, go, pv, lv))

            if epoch % plot_interval == 0:
                # Draw the decision boundary
                def draw_decision_boundary(x, w, y):
                    positive_x = x[y == 1, :]
                    negative_x = x[y == 0, :]

                    x0_points = np.linspace(lower_box, upper_box, num=100)
                    x1_points = np.linspace(lower_box, upper_box, num=100)
                    grid_x, grid_y = np.meshgrid(x0_points, x1_points)
                    points = np.vstack((np.ravel(grid_x), np.ravel(grid_y))).T
                    y_pred = 1.0 / (1 + np.exp(-np.sum(points * w, axis=1)))
                    grid_pred = np.reshape(y_pred, grid_x.shape)

                    plt.pcolormesh(grid_x, grid_y, grid_pred)
                    plt.colorbar()
                    CS = plt.contour(grid_x, grid_y, grid_pred, colors='k',
                                     levels=np.array([0.25, 0.5, 0.75]))
                    plt.clabel(CS)
                    plt.plot(positive_x[:, 0], positive_x[:, 1], 'x')
                    plt.plot(negative_x[:, 0], negative_x[:, 1], 'o')

                plt.subplot(3, 3, 1)
                draw_decision_boundary(train_x, train_w_sample, train_y)
                plt.title('Decision boundary')

                # Plot unnormalized true posterior
                # Generate a w grid
                w0 = np.linspace(lower_box, upper_box, 100)
                w1 = np.linspace(lower_box, upper_box, 100)
                w0_grid, w1_grid = np.meshgrid(w0, w1)
                w_points = np.vstack((np.ravel(w0_grid), np.ravel(w1_grid))).T
                # [n_particles]
                log_joint_points = sess.run(
                    log_joint_value,
                    feed_dict={x: train_x,
                               y: train_y,
                               n_particles: w_points.shape[0],
                               w_ph: w_points})
                log_joint_grid = np.reshape(log_joint_points, w0_grid.shape)

                plt.subplot(3, 3, 2)
                contourf(w0_grid, w1_grid, log_joint_grid)
                plt.plot(train_w_sample[0], train_w_sample[1], 'x')
                plt.title('Unnormalized true posterior')

                # Plot the gen/disc objectives
                plt.subplot(3, 3, 3)
                plt.plot(gen_objs, '.')
                plt.title('Generator objective')

                # Plot the variational posterior
                log_v_points = sess.run(
                    log_mean_field,
                    feed_dict={x: train_x,
                               y: train_y,
                               n_particles: w_points.shape[0],
                               w_ph: w_points})
                log_v_grid = np.reshape(log_v_points, w0_grid.shape)

                plt.subplot(3, 3, 4)
                contourf(w0_grid, w1_grid, log_v_grid)
                plt.title('Mean field posterior')

                # Plot samples from the implicit posterior
                samples = sess.run(qw_samples,
                                   feed_dict={n_particles: n_qw_samples})
                ax = plt.subplot(3, 3, 7)
                ax.scatter(samples[:, 0], samples[:, 1], s=0.1)
                ax.set_xlim(lower_box, upper_box)
                ax.set_ylim(lower_box, upper_box)
                plt.title('Implicit posterior samples')

                # Plot kde for the implicit posterior
                point_prob = np.zeros((w_points.shape[0]))
                kde_num_batches = n_qw_samples // kde_batch_size
                for b in range(kde_num_batches):
                    sample_batch = samples[
                        b * kde_batch_size:(b + 1) * kde_batch_size]
                    point_prob += sess.run(
                        grid_prob_op,
                        feed_dict={samples_ph: sample_batch,
                                   points_ph: w_points})
                point_prob /= kde_num_batches
                point_prob = np.reshape(point_prob, w0_grid.shape)

                plt.subplot(3, 3, 5)
                contourf(w0_grid, w1_grid, point_prob)
                plt.title('Implicit posterior KDE')

                plt.show()
                exit(0)
