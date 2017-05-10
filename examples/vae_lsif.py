#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import zhusuan as zs

from examples import conf
from examples.utils import dataset, save_image_collections


@zs.reuse('model')
def vae(observed, n, n_x, n_z, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_mean = tf.zeros([n, n_z])
        z_logstd = tf.zeros([n, n_z])
        z = zs.Normal('z', z_mean, z_logstd, n_samples=n_particles,
                      group_event_ndims=1)
        lx_z = layers.fully_connected(
            z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lx_z = layers.fully_connected(
            lx_z, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)
        x = zs.Bernoulli('x', x_logits, group_event_ndims=1)
    return model, z, x_logits


@zs.reuse('variational')
def q_net(observed, x, n_z, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        lz_x = layers.fully_connected(
            tf.to_float(x), 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        lz_x = layers.fully_connected(
            lz_x, 500, normalizer_fn=layers.batch_norm,
            normalizer_params=normalizer_params)
        z_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = zs.Normal('z', z_mean, z_logstd, n_samples=n_particles,
                      group_event_ndims=1)
    return variational, z_mean, z_logstd


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 1

    # Define training/evaluation parameters
    lb_samples = 100
    ll_samples = 1000
    epoches = 3000
    batch_size = 100
    iters = x_train.shape[0] // batch_size
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    save_freq = 100
    result_path = "results/vae_lsif"

    # LSIF parameters
    # kernel_width = 0.05
    lambda_ = 0.001
    n_basis = 100

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    # LSIF
    def rbf_kernel(z1, z2, kernel_width):
        return tf.exp(
            -tf.reduce_sum(tf.square(z1 - z2), -1) /
            (2 * tf.expand_dims(tf.expand_dims(kernel_width, -1), -1) ** 2))

    def phi(z, z_basis, kernel_width):
        # z: [n, particles, n_z]
        # z_basis: [n, n_basis, n_z]
        # phi(z): [n, n_particles, n_basis]
        z_row = tf.expand_dims(z, 2)
        z_col = tf.expand_dims(z_basis, 1)
        return rbf_kernel(z_row, z_col, kernel_width)

    def H(z, z_basis, kernel_width):
        # phi_z: [n, n_particles, n_basis]
        phi_z = phi(z, z_basis, kernel_width)
        # phi_z = tf.Print(phi_z, [phi_z], summarize=100)
        phi_z_t = tf.transpose(phi_z, perm=[0, 2, 1])
        # phi_z_t = tf.Print(phi_z_t, [tf.shape(phi_z), tf.shape(phi_z_t)],
        #                    message="phi_z:", summarize=20)
        # H: [n, n_basis, n_basis]
        return tf.matmul(phi_z_t, phi_z) / tf.to_float(tf.shape(z)[1])

    def h(z, z_basis, kernel_width):
        # h: [n, n_basis]
        return tf.reduce_mean(phi(z, z_basis, kernel_width), 1)

    def optimal_alpha(qz_samples, pz_samples, z_basis, kernel_width):
        # [n, n_basis, n_basis]
        H_ = H(pz_samples, z_basis, kernel_width)
        # [n, n_basis]
        h_ = h(qz_samples, z_basis, kernel_width)
        # H_ = tf.Print(H_, [H_], summarize=10000)
        alpha = tf.matmul(
            tf.matrix_inverse(H_ + lambda_ * tf.eye(tf.shape(H_)[1])),
            tf.expand_dims(h_, 2))
        # alpha: [n, n_basis]
        alpha = tf.squeeze(alpha, axis=-1)
        # alpha = tf.Print(alpha, [alpha], message="alpha: ", summarize=20)
        # alpha = tf.maximum(alpha, 0)
        return alpha

    def heuristic_kernel_width(z_samples, z_basis):
        n_z_samples = tf.shape(z_samples)[1]
        n_z_basis = tf.shape(z_basis)[1]
        z_samples = tf.expand_dims(z_samples, 2)
        z_basis = tf.expand_dims(z_basis, 1)
        pairwise_dist = tf.sqrt(
            tf.reduce_sum(tf.square(z_samples - z_basis), -1))
        k = n_z_samples * n_z_basis // 2
        top_k_values = tf.nn.top_k(
            tf.reshape(pairwise_dist,  [-1, n_z_samples * n_z_basis]),
            k=k).values
        kernel_width = top_k_values[:, -1]
        # kernel_width = tf.Print(kernel_width, [kernel_width],
        #                         message="kernel_width: ", summarize=20)
        return kernel_width

    def optimal_ratio(z, qz_samples, pz_samples):
        # z = tf.Print(z, [tf.shape(z), tf.shape(qz_samples), tf.shape(pz_samples)])
        # z, qz_samples, pz_samples: [n, n_particles, n_z]
        z_samples = tf.concat([qz_samples, pz_samples], axis=1)
        # z_basis: [n, n_basis, n_z]
        z_basis = qz_samples[:, :n_basis, :]
        # kernel_width: [n]
        kernel_width = heuristic_kernel_width(z_samples, z_basis)
        # alpha: [n, n_basis]
        alpha = optimal_alpha(qz_samples, pz_samples, z_basis, kernel_width)
        # phi_z: [n, N, n_basis]
        phi_z = phi(z, z_basis, kernel_width)
        ratio = tf.reduce_sum(tf.expand_dims(alpha, 1) * phi_z, -1)
        ratio = tf.maximum(ratio, 1e-16)
        # ratio: [n, N]
        # ratio = tf.Print(ratio, [ratio], message="ratio: ", summarize=20)
        return ratio

    variational, qz_mean, qz_logstd = q_net({}, x, n_z, n_particles, is_training)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    # qz_samples = tf.Print(qz_samples, [qz_mean, qz_logstd], "qz:",
    #                       summarize=20)
    # qz_samples = tf.Print(qz_samples, [qz_samples], message="qz_samples:",
    #                       summarize=20)
    observed = {'x': x_obs, 'z': qz_samples}
    model, z, _ = vae(observed, n, n_x, n_z, n_particles, is_training)
    log_pz, log_px_z = model.local_log_prob(['z', 'x'])
    ratio_t = tf.exp(log_pz - log_qz)
    # log_pz = tf.Print(log_pz, [tf.reduce_mean(ratio_t, 0)], message="ratio_t: ",
    #                   summarize=20)
    kl_term_t = tf.reduce_mean(log_qz - log_pz)
    # kl_term_t = tf.Print(kl_term_t, [log_qz], message="log_qz:", summarize=20)
    # kl_term_t = tf.Print(kl_term_t, [log_pz], message="log_pz:", summarize=20)
    eq_ll = tf.reduce_mean(log_px_z)

    pz_samples = z.sample(n_samples=n_particles)
    pz = tf.transpose(pz_samples, [1, 0, 2])
    qz = tf.transpose(qz_samples, [1, 0, 2])
    # kl_term = -tf.reduce_mean(
    #     tf.log(optimal_ratio(qz, pz, tf.stop_gradient(qz))))
    # kl_term = tf.reduce_mean(
    #     tf.log(optimal_ratio(qz, tf.stop_gradient(qz), tf.stop_gradient(pz))))
    ratio = optimal_ratio(qz, tf.stop_gradient(pz), tf.stop_gradient(qz))
    # ratio = tf.Print(ratio, [tf.reduce_mean(ratio, 1)], message="ratio: ",
    #                  summarize=20)
    kl_term = -tf.reduce_mean(tf.log(ratio))
    lower_bound = eq_ll - kl_term

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    # Generate images
    n_gen = 100
    _, _, x_logits = vae({}, n_gen, n_x, n_z, None, False)
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, epoches + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            kls = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
                # _, lb, kl, kl_t, px, qx, q_mean, q_logstd, est_r, true_r = sess.run(
                #     [infer, lower_bound, kl_term, kl_term_t,
                #      pz_samples, qz_samples, qz_mean, qz_logstd, ratio, ratio_t],
                _, lb, kl, kl_t = sess.run([infer, lower_bound, kl_term, kl_term_t],
                     feed_dict={x: x_batch_bin,
                                learning_rate_ph: learning_rate,
                                n_particles: lb_samples,
                                is_training: True})
                # print('Iter {}: kl = {}, kl_t = {}'.format(t, kl, kl_t))

                # print(px.shape, qx.shape)
                # print(q_mean.shape, q_logstd.shape)
                # print(est_r.shape, true_r.shape)
                # qx = qx.ravel()
                # px = px.ravel()
                # q_mean = q_mean.ravel()
                # q_std = np.exp(q_logstd).ravel()
                # true_r = true_r.ravel()
                # est_r = est_r.ravel()
                #
                # lower_box = -4
                # upper_box = 4
                #
                # def kde(xs, mu, batch_size, kde_stdev):
                #     mu_n = len(mu)
                #     assert mu_n % batch_size == 0
                #     xs_row = np.expand_dims(xs, 1)
                #     ys = np.zeros(xs.shape)
                #     for b in range(mu_n // batch_size):
                #         mu_col = np.expand_dims(
                #             mu[b * batch_size:(b + 1) * batch_size], 0)
                #         ys += (1 / np.sqrt(2 * np.pi) / kde_stdev) * \
                #               np.mean(np.exp((-0.5 / kde_stdev ** 2) *
                #                              np.square(xs_row - mu_col)), 1)
                #     ys /= (mu_n / batch_size)
                #     return ys
                #
                # # Plot 1: q, p distribution and samples
                # xs = np.linspace(-5, 5, 1000)
                # ax = plt.subplot(2, 1, 1)
                # q_curve = kde(xs, qx, lb_samples, 0.1)
                # p_curve = kde(xs, px, lb_samples, 0.1)
                # ax.plot(xs, q_curve)
                # ax.plot(xs, p_curve)
                # ax.plot(xs, stats.norm.pdf(xs, loc=q_mean, scale=q_std),
                #         label="q")
                # ax.plot(xs, stats.norm.pdf(xs, loc=0, scale=1), label="p")
                # ax.set_xlim(lower_box, upper_box)
                # ax.legend()
                #
                # # Plot 2: True ratio vs. estimated ratio (LSIF)
                # ax = plt.subplot(2, 1, 2)
                # idx = np.argsort(qx)
                # # True ratio
                # ax.plot(qx[idx], true_r[idx], '-o', label="True q/p",
                #         markersize=2)
                # # Estimated ratio (LSIF)
                # ax.plot(qx[idx], est_r[idx], '-o', label="Est. q/p (LSIF)",
                #         markersize=2)
                # ax.set_xlim(lower_box, upper_box)
                # ax.legend()
                #
                # plt.show()

                lbs.append(lb)
                kls.append(kl)

            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}, kl = {}'.format(
                epoch, time_epoch, np.mean(lbs), np.mean(kls)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples,
                                                  is_training: False})
                    test_lbs.append(test_lb)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))

            if epoch % save_freq == 0:
                print('Saving images...')
                images = sess.run(x_gen)
                name = os.path.join(result_path,
                                    "vae.epoch.{}.png".format(epoch))
                save_image_collections(images, name)

                print('Saving model...')
                save_path = os.path.join(result_path,
                                         "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                print('Done')
