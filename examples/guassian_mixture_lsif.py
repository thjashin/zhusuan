#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
from scipy import stats
import zhusuan as zs

# For Mac OS
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt


@zs.reuse("model")
def mixture_gaussian_1d(observed, n_particles):
    with zs.BayesianNet(observed) as model:
        z = zs.Bernoulli('z', 0., n_samples=n_particles, dtype=tf.float32)
        x = zs.Normal('x', z * (-3.) + (1. - z) * 3., 0.)
    return model


@zs.reuse("variational")
def mean_field_variational(n_particles):
    with zs.BayesianNet() as variational:
        x_mean = tf.get_variable('x_mean', shape=[], dtype=tf.float32,
                                 initializer=tf.constant_initializer(-5.))
        x_logstd = tf.get_variable('x_logstd', shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.))
        x = zs.Normal('x', x_mean, x_logstd, n_samples=n_particles)
    return variational, x_mean, x_logstd


if __name__ == "__main__":
    tf.set_random_seed(1235)
    np.random.seed(1234)

    # Define model parameters
    n_samples = 10000
    lower_box = -5
    upper_box = 5
    kde_batch_size = 2000
    learning_rate = 0.001

    # LSIF parameters
    # kernel_width = 0.2
    lambda_ = 0.1

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")

    def log_joint(observed):
        model = mixture_gaussian_1d(observed, n_particles)
        return model.local_log_prob('x')

    variational, x_mean, x_logstd = mean_field_variational(n_particles)
    qx_samples, log_qx = variational.query('x', outputs=True,
                                           local_log_prob=True)
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {}, {'x': [qx_samples, log_qx]}, axis=0))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    infer = optimizer.minimize(-lower_bound)

    # Set up figure.
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    # # Run the inference
    # iters = 1000
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for t in range(iters):
    #         _, lb, vmean, vlogstd = sess.run(
    #             [infer, lower_bound, x_mean, x_logstd],
    #             feed_dict={n_particles: 10})
    #         print('Iteration {}: lower bound = {}'.format(t, lb))
    #
    #         def draw(vmean, vlogstd):
    #             plt.cla()
    #             xs = np.linspace(-5, 5, 1000)
    #             true_curve = 0.5 * stats.norm.pdf(xs, -3., 1.) + \
    #                          0.5 * stats.norm.pdf(xs, 3., 1.)
    #             variational_curve = stats.norm.pdf(xs, vmean, np.exp(vlogstd))
    #             ax.plot(xs, true_curve)
    #             ax.plot(xs, variational_curve)
    #             ax.set_xlim(lower_box, upper_box)
    #             ax.set_ylim(0, 1)
    #             plt.draw()
    #             plt.pause(1. / 100000.)
    #
    #         draw(vmean, vlogstd)

    # LSIF
    def rbf_kernel(x1, x2, kernel_width):
        return tf.exp(-tf.square(x1 - x2) / (2 * kernel_width ** 2))

    def phi(w, w_basis, kernel_width):
        # w: [n_particles]
        # w_basis: [n_basis]
        # phi(w): [n_particles, n_basis]
        w_row = tf.expand_dims(w, 1)
        w_col = tf.expand_dims(w_basis, 0)
        return rbf_kernel(w_row, w_col, kernel_width)

    def H(w, w_basis, kernel_width):
        # phi_w: [n_particles, n_basis]
        phi_w = phi(w, w_basis, kernel_width)
        phi_w_t = tf.transpose(phi_w)
        # H: [n_basis, n_basis]
        return tf.matmul(phi_w_t, phi_w) / tf.to_float(tf.shape(w)[0])

    def h(w, w_basis, kernel_width):
        # h: [n_basis]
        return tf.reduce_mean(phi(w, w_basis, kernel_width), 0)

    def optimal_alpha(qw_samples, pw_samples, w_basis, kernel_width):
        H_ = H(pw_samples, w_basis, kernel_width)
        h_ = h(qw_samples, w_basis, kernel_width)
        K = phi(w_basis, w_basis, kernel_width)
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
        pairwise_dist = tf.abs(w_samples - w_basis)
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
        alpha = optimal_alpha(qw_samples, pw_samples, w_basis, kernel_width)
        # phi_x: [N, n_basis]
        phi_x = phi(x, w_basis, kernel_width)
        ratio = tf.reduce_sum(tf.expand_dims(alpha, 0) * phi_x, 1)
        # ratio = tf.maximum(0., ratio)
        # ratio: [N]
        return tf.Print(ratio, [ratio], message="ratio: ", summarize=20)

    @zs.reuse("implicit")
    def implicit_posterior(n_particles):
        with zs.BayesianNet():
            z = zs.Normal('z', [0.], 0., n_samples=n_particles)
            lx_z = layers.fully_connected(z, 10)
            x = layers.fully_connected(lx_z, 1)
            x = tf.squeeze(x, axis=1)
            return x

    model = mixture_gaussian_1d({}, n_particles)
    px_samples = model.outputs('x')
    qx_samples = implicit_posterior(n_particles)
    estimated_ratio = optimal_ratio(qx_samples, px_samples, qx_samples)
    implicit_lower_bound = tf.reduce_mean(tf.log(estimated_ratio))
    implicit_infer = optimizer.minimize(-implicit_lower_bound)

    # Run the inference
    iters = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t in range(iters):
            _, lb, samples = sess.run(
                [implicit_infer, implicit_lower_bound, qx_samples],
                feed_dict={n_particles: 100})
            print('Iteration {}: lower bound = {}'.format(t, lb))

            def kde(xs, mu, batch_size, kde_stdev):
                mu_n = len(mu)
                assert mu_n % batch_size == 0
                xs_row = np.expand_dims(xs, 1)
                ys = np.zeros(xs.shape)
                for b in range(mu_n // batch_size):
                    mu_col = np.expand_dims(
                        mu[b * batch_size:(b + 1) * batch_size], 0)
                    ys += (1 / np.sqrt(2 * np.pi) / kde_stdev) * \
                          np.mean(np.exp((-0.5 / kde_stdev ** 2) *
                                         np.square(xs_row - mu_col)), 1)
                ys /= (mu_n / batch_size)
                return ys

            def draw():
                plt.cla()
                xs = np.linspace(-5, 5, 1000)
                true_curve = 0.5 * stats.norm.pdf(xs, -3., 1.) + \
                             0.5 * stats.norm.pdf(xs, 3., 1.)
                q_curve = kde(xs, samples, 10, 0.1)
                ax.plot(xs, true_curve)
                ax.plot(xs, q_curve)
                ax.set_xlim(lower_box, upper_box)
                ax.set_ylim(0, 1)
                plt.draw()
                plt.pause(1. / 30.)

            draw()

    # Plotting
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     xs = np.linspace(-5, 5, 1000)
    #     true_r, est_r, qx, px = sess.run([true_ratio, estimated_ratio,
    #                                       qx_samples, px_samples],
    #                                      feed_dict={x_ph: xs})
    #     print("qx samples:", qx)
    #     print("true_r:", true_r)
    #
    #     def kde(xs, mu, batch_size, kde_stdev):
    #         mu_n = len(mu)
    #         assert mu_n % batch_size == 0
    #         xs_row = np.expand_dims(xs, 1)
    #         ys = np.zeros(xs.shape)
    #         for b in range(mu_n // batch_size):
    #             mu_col = np.expand_dims(
    #                 mu[b * batch_size:(b + 1) * batch_size], 0)
    #             ys += (1 / np.sqrt(2 * np.pi) / kde_stdev) * \
    #                   np.mean(np.exp((-0.5 / kde_stdev ** 2) *
    #                                  np.square(xs_row - mu_col)), 1)
    #         ys /= (mu_n / batch_size)
    #         return ys
    #
    #     # Plot 1: q, p distribution and samples
    #     ax = plt.subplot(2, 1, 1)
    #     q_curve = kde(xs, qx, 1000, 0.02)
    #     p_curve = kde(xs, px, 1000, 0.1)
    #     ax.plot(xs, q_curve)
    #     ax.plot(xs, p_curve)
    #     ax.plot(xs, stats.norm.pdf(xs, loc=q_mean, scale=q_std), label="q")
    #     ax.plot(xs, stats.norm.pdf(xs, loc=p_mean, scale=p_std), label="p")
    #     ax.set_xlim(lower_box, upper_box)
    #     ax.legend()
    #
    #     # Plot 2: True ratio vs. estimated ratio (LSIF)
    #     ax = plt.subplot(2, 1, 2)
    #     # normalized true ratio analytic
    #     # r_mean = (q_mean * q_precision - p_mean * p_precision) / (
    #     #     q_precision - p_precision)
    #     # r_precision = q_precision - p_precision
    #     # r_std = np.sqrt(1. / r_precision)
    #     # true_r_analytic = stats.norm.pdf(xs, loc=r_mean, scale=r_std)
    #     # ax.plot(xs, true_r_analytic, label="Normalized true q/p")
    #     # True ratio
    #     ax.plot(xs, true_r, label="True q/p")
    #     # Estimated ratio (LSIF)
    #     ax.plot(xs, est_r, label="Est. q/p (LSIF)")
    #     ax.set_xlim(lower_box, upper_box)
    #     ax.legend()
    #
    #     plt.show()
