#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from six.moves import range
import numpy as np
from scipy import stats
import zhusuan as zs

# For Mac OS
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import cm, colors


q_mean = 2.
q_logstd = -1.5
q_std = np.exp(q_logstd)
q_precision = 1. / q_std ** 2
p_mean = 1.
p_logstd = -0.5
p_std = np.exp(p_logstd)
p_precision = 1. / p_std ** 2


@zs.reuse('q')
def q(observed, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        x = zs.Normal('x', q_mean, q_logstd, n_samples=n_particles)
    return model


@zs.reuse('p')
def p(observed, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        x = zs.Normal('x', p_mean, p_logstd, n_samples=n_particles)
    return model


if __name__ == "__main__":
    tf.set_random_seed(1235)
    np.random.seed(1234)

    # Define model parameters
    N = 200
    D = 2
    n_samples = 1000
    lower_box = -1
    upper_box = 4
    kde_batch_size = 2000

    # LSIF parameters
    kernel_width = 0.2
    lambda_ = 0.2

    # Build the computation graph
    x_ph = tf.placeholder(tf.float32, [None])

    # Generate synthetic data
    q_model = q({}, n_samples)
    p_model = p({}, n_samples)
    qx_samples, log_qx = q_model.query('x', outputs=True, local_log_prob=True)
    px_samples, log_px = p_model.query('x', outputs=True, local_log_prob=True)

    # Plotting computation
    p_model_obs = p({'x': x_ph}, n_samples)
    q_model_obs = q({'x': x_ph}, n_samples)
    log_p_given_x = p_model_obs.local_log_prob('x')
    log_q_given_x = q_model_obs.local_log_prob('x')
    true_log_ratio = log_q_given_x - log_p_given_x
    true_ratio = tf.exp(true_log_ratio)

    # LSIF
    def rbf_kernel(x1, x2):
        return tf.exp(-tf.square(x1 - x2) / (2 * kernel_width ** 2))

    def phi(w, w_basis):
        # w: [n_particles]
        # w_basis: [n_basis]
        # phi(w): [n_particles, n_basis]
        w_row = tf.expand_dims(w, 1)
        w_col = tf.expand_dims(w_basis, 0)
        return rbf_kernel(w_row, w_col)

    def H(w, w_basis):
        # phi_w: [n_particles, n_basis]
        phi_w = phi(w, w_basis)
        phi_w_t = tf.transpose(phi_w)
        # H: [n_basis, n_basis]
        return tf.matmul(phi_w_t, phi_w) / tf.to_float(tf.shape(w)[0])
        # phi_w = phi(w, w_basis)
        # return tf.reduce_mean(
        #     tf.expand_dims(phi_w, 2) * tf.expand_dims(phi_w, 1), 0)

    def h(w, w_basis):
        # h: [n_basis]
        return tf.reduce_mean(phi(w, w_basis), 0)

    def optimal_alpha(qw_samples, pw_samples):
        H_ = H(pw_samples, qw_samples)
        h_ = h(qw_samples, qw_samples)
        alpha = tf.matmul(
            tf.matrix_inverse(H_ + lambda_ * tf.eye(tf.shape(H_)[0])),
            tf.expand_dims(h_, 1))
        # alpha: [n_basis] = [n_particles]
        alpha = tf.squeeze(alpha, axis=1)
        alpha = tf.Print(alpha, [alpha], message="alpha: ", summarize=20)
        # alpha = tf.maximum(alpha, 0)
        return alpha

    def optimal_ratio(x, qw_samples, pw_samples):
        alpha = optimal_alpha(qw_samples, pw_samples)
        # phi_x: [N, n_basis]
        phi_x = phi(x, qw_samples)
        ratio = tf.reduce_sum(tf.expand_dims(alpha, 0) * phi_x, 1)
        # ratio = tf.maximum(0., ratio)
        # ratio: [N]
        return tf.Print(ratio, [ratio], summarize=20)

    # estimated_ratio: [N]
    estimated_ratio = optimal_ratio(x_ph, qx_samples, px_samples)

    # Plotting
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        xs = np.linspace(-5, 5, 1000)
        true_r, est_r, qx, px = sess.run([true_ratio, estimated_ratio,
                                          qx_samples, px_samples],
                                         feed_dict={x_ph: xs})
        print("qx samples:", qx)
        print("true_r:", true_r)

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

        # Plot 1: q, p distribution and samples
        ax = plt.subplot(2, 1, 1)
        q_curve = kde(xs, qx, 1000, 0.02)
        p_curve = kde(xs, px, 1000, 0.1)
        ax.plot(xs, q_curve)
        ax.plot(xs, p_curve)
        ax.plot(xs, stats.norm.pdf(xs, loc=q_mean, scale=q_std), label="q")
        ax.plot(xs, stats.norm.pdf(xs, loc=p_mean, scale=p_std), label="p")
        ax.set_xlim(lower_box, upper_box)
        ax.legend()

        # Plot 2: True ratio vs. estimated ratio (LSIF)
        ax = plt.subplot(2, 1, 2)
        # normalized true ratio analytic
        r_mean = (q_mean * q_precision - p_mean * p_precision) / (
            q_precision - p_precision)
        r_precision = q_precision - p_precision
        r_std = np.sqrt(1. / r_precision)
        true_r_analytic = stats.norm.pdf(xs, loc=r_mean, scale=r_std)
        ax.plot(xs, true_r_analytic, label="Normalized true q/p")
        # True ratio
        ax.plot(xs, true_r, label="True q/p")
        # Estimated ratio (LSIF)
        ax.plot(xs, est_r, label="Est. q/p (LSIF)")
        ax.set_xlim(lower_box, upper_box)
        ax.legend()

        plt.show()
