#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import six
from six.moves import zip, map
from tensorflow.python.training import moving_averages

from .utils import log_mean_exp, merge_dicts
from .evaluation import is_loglikelihood


__all__ = [
    'advi',
    'iwae',
    'rws',
    'nvil',
    'vimco',
    'gan',
    'ali',
    'avb'
]


def advi(log_joint, observed, latent, axis=0):
    """
    Implements the automatic differentiation variational inference (ADVI)
    algorithm. This only works for continuous latent `StochasticTensor` s that
    can be reparameterized (Kingma, 2013).

    :param log_joint: A function that accepts a dictionary argument of
        (str, Tensor) pairs, which are mappings from all `StochasticTensor`
        names in the model to their observed values. The function should
        return a Tensor, representing the log joint likelihood of the model.
    :param observed: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values
    :param latent: A dictionary of (str, (Tensor, Tensor)) pairs. Mapping
        from names of latent `StochasticTensor` s to their samples and log
        probabilities.
    :param axis: The sample dimension(s) to reduce when computing the
        variational lower bound.

    :return: A Tensor. The variational lower bound.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    joint_obs = merge_dicts(observed, latent_outputs)
    lower_bound = log_joint(joint_obs) - sum(latent_logpdfs)
    lower_bound = tf.reduce_mean(lower_bound, axis)
    return lower_bound


def iwae(log_joint, observed, latent, axis=0):
    """
    Implements the importance weighted lower bound from (Burda, 2015).
    This only works for continuous latent `StochasticTensor` s that
    can be reparameterized (Kingma, 2013).

    :param log_joint: A function that accepts a dictionary argument of
        (str, Tensor) pairs, which are mappings from all `StochasticTensor`
        names in the model to their observed values. The function should
        return a Tensor, representing the log joint likelihood of the model.
    :param observed: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values.
    :param latent: A dictionary of (str, (Tensor, Tensor)) pairs. Mapping
        from names of latent `StochasticTensor` s to their samples and log
        probabilities.
    :param axis: The sample dimension(s) to reduce when computing the
        variational lower bound.

    :return: A Tensor. The importance weighted lower bound.
    """
    return is_loglikelihood(log_joint, observed, latent, axis)


def rws(log_joint, observed, latent, axis=0):
    """
    Implements Reweighted Wake-sleep from (Bornschein, 2015). This works for
    both continuous and discrete latent `StochasticTensor` s.

    :param log_joint: A function that accepts a dictionary argument of
        (str, Tensor) pairs, which are mappings from all `StochasticTensor`
        names in the model to their observed values. The function should
        return a Tensor, representing the log joint likelihood of the model.
    :param observed: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values.
    :param latent: A dictionary of (str, (Tensor, Tensor)) pairs. Mapping
        from names of latent `StochasticTensor` s to their samples and log
        probabilities.
    :param axis: The sample dimension(s) to reduce when computing the
        log likelihood and the cost for adapting proposals.

    :return: A Tensor. The surrogate cost to minimize.
    :return: A Tensor. Estimated log likelihoods.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    joint_obs = merge_dicts(observed, latent_outputs)
    log_joint_value = log_joint(joint_obs)
    entropy = -sum(latent_logpdfs)
    log_w = log_joint_value + entropy
    log_w_max = tf.reduce_max(log_w, axis, keep_dims=True)
    w_u = tf.exp(log_w - log_w_max)
    w_tilde = tf.stop_gradient(w_u / tf.reduce_sum(w_u, axis, keep_dims=True))
    log_likelihood = log_mean_exp(log_w, axis)
    fake_log_joint_cost = -tf.reduce_sum(w_tilde * log_joint_value, axis)
    fake_proposal_cost = tf.reduce_sum(w_tilde * entropy, axis)
    cost = fake_log_joint_cost + fake_proposal_cost
    return cost, log_likelihood


def nvil(log_joint,
         observed,
         latent,
         baseline=None,
         decay=0.8,
         variance_normalization=False,
         axis=0):
    """
    Implements the variance reduced score function estimator for gradients
    of the variational lower bound from (Mnih, 2014). This algorithm is also
    called "REINFORCE" or "baseline". This works for both continuous and
    discrete latent `StochasticTensor` s.

    :param log_joint: A function that accepts a dictionary argument of
        (str, Tensor) pairs, which are mappings from all `StochasticTensor`
        names in the model to their observed values. The function should
        return a Tensor, representing the log joint likelihood of the model.
    :param observed: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values.
    :param latent: A dictionary of (str, (Tensor, Tensor)) pairs. Mapping
        from names of latent `StochasticTensor` s to their samples and log
        probabilities.
    :param baseline: A Tensor with the same shape as returned by `log_joint`.
        A trainable estimation for the scale of the variational lower bound,
        which is typically dependent on observed values, e.g., a neural
        network with observed values as inputs.
    :param variance_normalization: Whether to use variance normalization.
    :param decay: Float. The moving average decay for variance normalization.
    :param axis: The sample dimension(s) to reduce when computing the
        variational lower bound.

    :return: A Tensor. The surrogate cost to minimize.
    :return: A Tensor. The variational lower bound.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    joint_obs = merge_dicts(observed, latent_outputs)
    log_joint_value = log_joint(joint_obs)
    entropy = -sum(latent_logpdfs)
    l_signal = log_joint_value + entropy
    cost = 0.

    if baseline is not None:
        baseline = tf.expand_dims(baseline, axis)
        baseline_cost = 0.5 * tf.reduce_mean(tf.square(
            tf.stop_gradient(l_signal) - baseline), axis)
        l_signal = l_signal - baseline
        cost += baseline_cost

    if variance_normalization is True:
        bc = tf.reduce_mean(l_signal)
        bv = tf.reduce_mean(tf.square(l_signal - bc))
        moving_mean = tf.get_variable(
            'moving_mean', shape=[], initializer=tf.constant_initializer(0.),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', shape=[],
            initializer=tf.constant_initializer(1.), trainable=False)

        update_mean = moving_averages.assign_moving_average(
            moving_mean, bc, decay=decay)
        update_variance = moving_averages.assign_moving_average(
            moving_variance, bv, decay=decay)
        l_signal = (l_signal - moving_mean) / tf.maximum(
            1., tf.sqrt(moving_variance))
        with tf.control_dependencies([update_mean, update_variance]):
            l_signal = tf.identity(l_signal)

    fake_log_joint_cost = -tf.reduce_mean(log_joint_value, axis)
    fake_variational_cost = tf.reduce_mean(
        tf.stop_gradient(l_signal) * entropy, axis)
    cost += fake_log_joint_cost + fake_variational_cost
    lower_bound = tf.reduce_mean(log_joint_value + entropy, axis)
    return cost, lower_bound


def vimco(log_joint, observed, latent, axis=0):
    """
    Implements the multi-sample variance reduced score function estimator for
    gradients of the variational lower bound from (Minh, 2016). This works for
    both continuous and discrete latent `StochasticTensor` s.

    :param log_joint: A function that accepts a dictionary argument of
        (str, Tensor) pairs, which are mappings from all `StochasticTensor`
        names in the model to their observed values. The function should
        return a Tensor, representing the log joint likelihood of the model.
    :param observed: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values.
    :param latent: A dictionary of (str, (Tensor, Tensor)) pairs. Mapping
        from names of latent `StochasticTensor` s to their samples and log
        probabilities.
    :param axis: The sample dimension to reduce when computing the
        variational lower bound.

    :return: A Tensor. The surrogate cost to minimize.
    :return: A Tensor. The variational lower bound.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    joint_obs = merge_dicts(observed, latent_outputs)
    log_joint_value = log_joint(joint_obs)
    entropy = -sum(latent_logpdfs)
    l_signal = log_joint_value + entropy

    # check ndim of sample axis
    static_signal_shape = l_signal.get_shape()
    if static_signal_shape[axis:axis+1].is_fully_defined():
        K = int(static_signal_shape[axis])
        if K < 2:
            raise ValueError('ndim of sample axis should be larger than 1')
    dynamic_signal_shape = tf.shape(l_signal)
    _assert_axis_dim = tf.assert_greater_equal(dynamic_signal_shape[axis], 2,
                                               message="ndim of sample axis should be larger than 1")
    with tf.control_dependencies([_assert_axis_dim]):
        l_signal = tf.identity(l_signal)

    # compute variance reduction term
    mean_except_signal = (tf.reduce_sum(l_signal, axis, keep_dims=True) -
        l_signal) / tf.to_float(tf.shape(l_signal)[axis] - 1)
    x, sub_x = tf.to_float(l_signal), tf.to_float(mean_except_signal)

    n_dim = tf.rank(x)
    axis_dim_mask = tf.cast(tf.one_hot(axis, n_dim), tf.bool)
    original_mask = tf.cast(tf.one_hot(n_dim - 1, n_dim), tf.bool)
    axis_dim = tf.ones([n_dim], tf.int32) * axis
    originals = tf.ones([n_dim], tf.int32) * (n_dim - 1)
    perm = tf.where(original_mask, axis_dim, tf.range(n_dim))
    perm = tf.where(axis_dim_mask, originals, perm)
    multiples = tf.concat([tf.ones([n_dim], tf.int32), [tf.shape(x)[axis]]], 0)

    x = tf.transpose(x, perm=perm)
    sub_x = tf.transpose(sub_x, perm=perm)
    x_ex = tf.tile(tf.expand_dims(x, n_dim), multiples)
    x_ex = x_ex - tf.matrix_diag(x) + tf.matrix_diag(sub_x)
    pre_signal = tf.transpose(log_mean_exp(x_ex, n_dim - 1), perm=perm)

    # variance reduced objective
    l_signal = log_mean_exp(l_signal, axis, keep_dims=True) - pre_signal
    fake_term = tf.reduce_sum(-entropy * tf.stop_gradient(l_signal), axis)
    lower_bound = log_mean_exp(log_joint_value + entropy, axis)
    cost = -fake_term - log_mean_exp(log_joint_value + entropy, axis)

    return cost, lower_bound


def gan(classifier, observed, gen_obs):
    """
    Implements the generative adversarial nets (GAN) algorithm. This only
    works for continuous latent `StochasticTensor` s that can be
    reparameterized (Kingma, 2013).

    :param classifier: A function that accepts a Tensor. The function should
        return a Tensor or several Tensors, representing the unnormalized log
        probability that the corresponding latent input is real instead of
        being generated by gen_graph.
    :param observed: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values.
    :param gen_obs: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values.

    :return disc_loss: A Tensor. The loss to be minimized by discriminator.
    :return gen_loss: A Tensor. The loss to be minimized by generator.
    """
    real_class_logits = classifier(observed)
    gen_class_logits = classifier(gen_obs)

    if not isinstance(real_class_logits, type((0,))):
        real_class_logits = [real_class_logits]
        gen_class_logits = [gen_class_logits]
    real_class_logits = list(real_class_logits)
    gen_class_logits = list(gen_class_logits)

    disc_loss = sum([
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(rl), logits=rl)) +
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(gl), logits=gl))
        for rl, gl in zip(real_class_logits, gen_class_logits)]) / 2.

    gen_loss = sum([
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(gl), logits=gl))
        for gl in gen_class_logits])

    return disc_loss, gen_loss


def ali(classifier, encoder, decoder):
    """
    Implements the Adversarial Learned Inference (ALI) algorithm. This only
    works for continuous latent `StochasticTensor` s that can be
    reparameterized (Kingma, 2013).

    :param classifier: A function that accepts a dict of (str, Tensor),
        representing the mapping from tensor's name to value. The function
        should return a Tensor or several Tensors, representing the
        unnormalized log probability that the corresponding input is
        generated by encoder instead of being generated by decoder.
    :param encoder: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values.
    :param decoder: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values.

    :return disc_loss: A Tensor. The loss to be minimized by discriminator.
    :return gen_loss: A Tensor. The loss to be minimized by generator.
    """
    enc_class_logits = classifier(encoder)
    dec_class_logits = classifier(decoder)

    if not isinstance(enc_class_logits, type((0,))):
        enc_class_logits = [enc_class_logits]
        dec_class_logits = [dec_class_logits]
    enc_class_logits = list(enc_class_logits)
    dec_class_logits = list(dec_class_logits)

    disc_loss = sum([
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(el), logits=el)) +
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(dl), logits=dl))
        for el, dl in zip(enc_class_logits, dec_class_logits)])
    gen_loss = sum([
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(el), logits=el)) +
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(dl), logits=dl))
        for el, dl in zip(enc_class_logits, dec_class_logits)])
    return disc_loss, gen_loss


def avb(log_like, classifier, observed, latent, prior):
    """
    Implements the Adversarial Variational Bayes (AVB) algorithm. This
    only works for continuous latent `StochasticTensor` s that can be
    reparameterized (Kingma, 2013).

    :param log_like: A function that accepts a dictionary argument of
        (str, Tensor) pairs, which are mappings from all `StochasticTensor`
        names in the model to their observed values. The function should
        return a Tensor, representing the log conditional likelihood of the
        model.
    :param classifier: A function that accepts a dict of (str, Tensor),
        representing the mapping from tensor's name to value. The function
        should return a Tensor or several Tensors, representing the
        unnormalized log probability that the corresponding input is generated
        by encoder instead of being generated by decoder.
    :param observed: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values
    :param latent: A dictionary of (str, Tensor) pairs. Mapping
        from names of latent `StochasticTensor` s to their samples,
        representing the latent variables generated by inference.
    :param prior: A dictionary of (str, Tensor) pairs. Mapping
        from names of latent `StochasticTensor` s to their samples,
        representing the latent variables generated by prior.

    :return model_loss: A Tensor. The loss to be minimized by model parameters.
    :return infer_loss: A Tensor. The loss to be minimized by inference
        parameters.
    :return disc_loss: A Tensor. The loss to be minimized by discriminator
        parameters.
    """
    infer_class_logits = classifier(observed, latent)
    prior_class_logits = classifier(observed, prior)

    if not isinstance(infer_class_logits, type((0,))):
        infer_class_logits = [infer_class_logits]
        prior_class_logits = [prior_class_logits]
    infer_class_logits = list(infer_class_logits)
    prior_class_logits = list(prior_class_logits)

    joint_obs = merge_dicts(observed, latent)
    model_loss = -tf.reduce_mean(log_like(joint_obs))
    infer_loss = model_loss + sum([tf.reduce_mean(il)
                                   for il in infer_class_logits])
    disc_loss = sum([
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(il), logits=il)) +
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(pl), logits=pl))
        for il, pl in zip(infer_class_logits, prior_class_logits)])

    return model_loss, infer_loss, disc_loss, infer_class_logits, prior_class_logits
