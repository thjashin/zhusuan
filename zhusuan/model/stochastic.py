#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from zhusuan.utils import add_name_scope
from zhusuan.distributions import norm, discrete, bernoulli
from .base import StochasticTensor


__all__ = [
    'Normal',
    'Bernoulli',
    'Discrete',
]


class Normal(StochasticTensor):
    """
    The class of independent Normal StochasticTensor.

    :param mean: A Tensor, python value, or numpy array. The mean of the
        Normal distribution.
    :param logstd: A Tensor, python value, or numpy array. The log
        standard deviation of the Normal distribution. Should have the same
        shape with `mean`.
    :param sample_dim: A Tensor scalar, int, or None. The sample dimension.
        If None, this means no new sample dimension is created. In this
        case `n_samples` must be set to 1, otherwise an Exception is
        raised.
    :param n_samples: A Tensor scalar or int. Number of samples to
        generate.
    :param reparameterized: Bool. If True, gradients on samples from this
        Normal distribution are allowed to propagate into inputs in this
        function, using the reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """
    def __init__(self,
                 mean,
                 logstd,
                 sample_dim=None,
                 n_samples=1,
                 reparameterized=True,
                 check_numerics=True):
        incomings = [mean, logstd, sample_dim, n_samples]
        self.reparameterized = reparameterized
        self.check_numerics = check_numerics
        super(Normal, self).__init__(incomings)

    @add_name_scope
    def sample(self, **kwargs):
        mean, logstd, sample_dim, n_samples = self.incomings
        return norm.rvs(mean=mean,
                        logstd=logstd,
                        sample_dim=sample_dim,
                        n_samples=n_samples,
                        reparameterized=self.reparameterized,
                        check_numerics=self.check_numerics)

    @add_name_scope
    def log_prob(self, given, inputs):
        mean, logstd, sample_dim, n_samples = inputs
        return norm.logpdf(given,
                           mean=mean,
                           logstd=logstd,
                           sample_dim=sample_dim,
                           check_numerics=self.check_numerics)


class Bernoulli(StochasticTensor):
    """
    The class of independent Bernoulli StochasticTensor.

    :param logits: A Tensor, python value, or numpy array. The unnormalized
        log probabilities of being 1. (:math:`logits=\log \frac{p}{1 - p}`)
    :param sample_dim: A Tensor scalar, int, or None. The sample dimension.
        If None, this means no new sample dimension is created. In this
        case `n_samples` must be set to 1, otherwise an Exception is
        raised.
    :param n_samples: A Tensor scalar or int. Number of samples to
        generate.
    """
    def __init__(self, logits, sample_dim=None, n_samples=1):
        incomings = [logits, sample_dim, n_samples]
        super(Bernoulli, self).__init__(incomings)

    @add_name_scope
    def sample(self, **kwargs):
        logits, sample_dim, n_samples = self.incomings
        return bernoulli.rvs(logits,
                             sample_dim=sample_dim,
                             n_samples=n_samples)

    @add_name_scope
    def log_prob(self, given, inputs):
        logits, sample_dim, n_samples = inputs
        return bernoulli.logpmf(given, logits, sample_dim=sample_dim)


class Discrete(StochasticTensor):
    """
    The class of Discrete StochasticTensor.

    :param logits: A N-D (N >= 1) Tensor, python value, or numpy array of shape
        (..., n_classes). Each slice `[i, j,..., k, :]`
        represents the un-normalized log probabilities for all classes.
    :param sample_dim: A Tensor scalar, int, or None. The sample dimension.
        If None, this means no new sample dimension is created. In this
        case `n_samples` must be set to 1, otherwise an Exception is
        raised.
    :param n_samples: A Tensor scalar or int. Number of samples to
        generate.

    The sample is a N-D or (N+1)-D Tensor of shape
    (..., [n_samples,] ..., n_classes). Each slice is a one-hot vector
    of the sample.
    """
    def __init__(self, logits, sample_dim=None, n_samples=1):
        incomings = [logits, sample_dim, n_samples]
        super(Discrete, self).__init__(incomings)

    @add_name_scope
    def sample(self, **kwargs):
        logits, sample_dim, n_samples = self.incomings
        return discrete.rvs(logits,
                            sample_dim=sample_dim,
                            n_samples=n_samples)

    @add_name_scope
    def log_prob(self, given, inputs):
        logits, sample_dim, n_samples = inputs
        return discrete.logpmf(given, logits, sample_dim=sample_dim)
