#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']="3"
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zhusuan as zs

import utils
import dataset
import logging


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
    return model


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
        lz_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
        lz_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
        z = zs.Normal('z', lz_mean, lz_logstd, n_samples=n_particles,
                      group_event_ndims=1)
    return variational


@zs.reuse('discriminator')
def discriminator(observed, latent):
    """
    :param observed: dict of {'x': x}. The shape of x should be of
        (n_particles, batch_size, shape_x).
    :param latent: dict of {'z':z}. The shape of z should be of
        (n_particles, batch_size, shape_z).

    :return: tensor of shape (n_particles, batch_size).
    """
    x = observed['x']
    z = latent['z']
    lc_z = layers.fully_connected(z, 500)
    lc_z = layers.fully_connected(lc_z, 500)

    lc_x = tf.to_float(x)
    lc_x = layers.fully_connected(lc_x, 500)
    lc_x = layers.fully_connected(lc_x, 500)

    lc_xz = tf.concat([lc_z, lc_x], 2)
    lc_xz = layers.fully_connected(lc_xz, 1000)
    lc_xz = layers.fully_connected(lc_xz, 1, activation_fn=None)
    return lc_xz


if __name__ == "__main__":
    tf.set_random_seed(1237)

    logger = logging.getLogger('avb_vae')
    logger.setLevel(logging.DEBUG)
    info_file_handler = logging.FileHandler('logs/avb_vae/1.log')
    info_file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(info_file_handler)
    logger.addHandler(console_handler)

    # Load MNIST
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_z = 40

    # Define training/evaluation parameters
    lb_samples = 1
    ll_samples = 1000
    epoches = 3000
    batch_size = 100
    iters = x_train.shape[0] // batch_size
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    test_freq = 5
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    test_n_temperatures = 100
    test_n_leapfrogs = 10
    test_n_chains = 10
    save_freq = 100
    result_path = "results/vae"
    recon_size = 50

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    def log_joint(observed):
        model = vae(observed, n, n_x, n_z, n_particles, is_training)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    def log_like(observed):
        model = vae(observed, n, n_x, n_z, n_particles, is_training)
        log_px_z = model.local_log_prob(['x'])
        return log_px_z

    variational = q_net({}, x, n_z, n_particles, is_training)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)

    prior_model = vae({}, n, n_x, n_z, n_particles, is_training)
    pz_samples = prior_model.outputs('z')

    model_loss, q_loss, disc_loss, _, _ = zs.avb(
        log_like, discriminator, {'x': x_obs}, {'z': qz_samples},
        {'z': pz_samples})

    # Importance sampling estimates of log likelihood:
    # Fast, used for evaluation during training
    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x_obs},
                            {'z': [qz_samples, log_qz]}, axis=0))

    # Bidirectional Monte Carlo (BDMC) estimates of log likelihood:
    # Slower than IS estimates, used for evaluation after training
    # Use q(z|x) as prior in BDMC
    def log_qz_given_x(observed):
        z = observed['z']
        model = q_net({'z': z}, x, n_z, n_particles, is_training)
        return model.local_log_prob('z')

    prior_samples = {'z': qz_samples}
    z = tf.Variable(tf.zeros([test_n_chains, test_batch_size, n_z]),
                    name="z", trainable=False)
    hmc = zs.HMC(step_size=1e-6, n_leapfrogs=test_n_leapfrogs,
                 adapt_step_size=True, target_acceptance_rate=0.65,
                 adapt_mass=True)
    bdmc = zs.BDMC(log_qz_given_x, log_joint, prior_samples, hmc,
                   {'x': x_obs}, {'z': z},
                   n_chains=test_n_chains, n_temperatures=test_n_temperatures)

    # Reconstruction of Picture for evalution
    recon_model = vae({'z': qz_samples}, n, n_x, n_z,
                      n_particles, is_training)
    x_recon = recon_model.outputs('x')
    x_recon_smooth = tf.reduce_mean(tf.to_float(x_recon), 0)
    x_recon_sharp = tf.to_float(tf.greater(x_recon_smooth, 0.5))

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)

    model_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
    q_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='variational')
    disc_var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    model_grads = optimizer.compute_gradients(model_loss,
                                              var_list=model_var_list)
    q_grads = optimizer.compute_gradients(q_loss, var_list=q_var_list)
    disc_grads = optimizer.compute_gradients(disc_loss, var_list=disc_var_list)

    grads = model_grads + q_grads + disc_grads
    infer = optimizer.apply_gradients(grads)

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
            mls, qls, dls = [], [], []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
                _, ml, ql, dl = sess.run(
                    [infer, model_loss, q_loss, disc_loss],
                    feed_dict={x: x_batch_bin,
                               learning_rate_ph: learning_rate,
                               n_particles: lb_samples,
                               is_training: True})
                mls.append(ml)
                qls.append(ql)
                dls.append(dl)
            time_epoch += time.time()
            logger.info('Epoch {} ({:.1f}s):'.format(epoch, time_epoch))
            logger.info(
                'Model loss = {}, Inference loss = {},'
                ' Disc loss = {}'.format(
                np.mean(mls), np.mean(qls), np.mean(dls)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_mls, test_qls, test_dls = [], [], []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_ml, test_ql, test_dl = sess.run(
                        [model_loss, q_loss, disc_loss],
                        feed_dict={x: test_x_batch, n_particles: lb_samples,
                                   is_training: False})
                    test_ll = sess.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  is_training: False})
                    test_mls.append(test_ml)
                    test_qls.append(test_ql)
                    test_dls.append(test_dl)
                    test_lls.append(test_ll)
                time_test += time.time()
                logger.info('>>> TEST ({:.1f}s)'.format(time_test))
                logger.info(
                    '>> Test model loss = {}, infer loss = {},'
                    ' disc loss = {}'.format(
                    np.mean(test_mls), np.mean(test_qls), np.mean(dls)))
                logger.info('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))

                # image reconstruction
                per = np.random.permutation(x_test.shape[0])
                recon_x_batch = x_test[per[:recon_size]]
                recon_images_smooth, recon_images_sharp = sess.run(
                    [x_recon_smooth, x_recon_sharp],
                    feed_dict={x: recon_x_batch, n_particles: ll_samples,
                               is_training: False})
                x_batch_images = np.reshape(recon_x_batch, [-1, 28, 28, 1])
                recon_images_smooth = np.reshape(recon_images_smooth,
                                                 [-1, 28, 28, 1])
                recon_images_sharp = np.reshape(recon_images_sharp,
                                                [-1, 28, 28, 1])
                name_smooth = "results/avb_vae/recon/'\
                    'smooth_avb_vae.epoch.{}.png".format(epoch)
                name_sharp = "results/avb_vae/recon/'\
                    'sharp_avb_vae.epoch.{}.png".format(epoch)
                utils.save_contrast_image_collections(
                    x_batch_images, recon_images_smooth,
                    name_smooth, scale_each=True)
                utils.save_contrast_image_collections(
                    x_batch_images, recon_images_sharp,
                    name_sharp, scale_each=True)

            if epoch % save_freq == 0:
                logger.info('Saving model...')
                save_path = os.path.join(result_path,
                                         "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                logger.info('Done')

        # BDMC evaluation
        logger.info('Start evaluation...')
        time_bdmc = -time.time()
        test_ll_lbs = []
        test_ll_ubs = []
        for t in range(test_iters):
            test_x_batch = x_test[t * test_batch_size:
                                  (t + 1) * test_batch_size]
            ll_lb, ll_ub = bdmc.run(
                sess,
                feed_dict={x: test_x_batch, n_particles: test_n_chains,
                           is_training: False})
            test_ll_lbs.append(ll_lb)
            test_ll_ubs.append(ll_ub)
        time_bdmc += time.time()
        test_ll_lb = np.mean(test_ll_lbs)
        test_ll_ub = np.mean(test_ll_ubs)
        logger.info('>> Test log likelihood (BDMC) ({:.1f}s)\n'
                    '>> lower bound = {}, upper bound = {}, BDMC gap = {}'
                    .format(time_bdmc, test_ll_lb, test_ll_ub,
                            test_ll_ub - test_ll_lb))
