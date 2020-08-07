#!/usr/bin/env python3

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    X is a numpy.ndarray of shape (n, d)
     containing the data set
    pi is a numpy.ndarray of shape (k,)
     containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d)
     containing the centroid means for each cluster
    S is a numpy.ndarray of shape (k, d, d)
     containing the covariance matrices for each cluster
    You may use at most 1 loop
    Returns: g, l, or None, None on failure
    g is a numpy.ndarray of shape (k, n)
     containing the posterior probabilities
      for each data point in each cluster
    l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if X.shape[1] != m.shape[1] or X.shape[1] != S.shape[1]:
        return None, None
    if X.shape[1] != S.shape[2]:
        return None, None
    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    n, d = X.shape
    k = pi.shape[0]
    g_sum = 0
    g = np.zeros((k, n))
    for ki in range(k):
        gi = pi[ki] * pdf(X, m[ki], S[ki])
        g[ki] = gi
        g_sum = g_sum + gi
    g = g / g_sum
    log_likelihood = np.sum(np.log(g_sum))
    return g, log_likelihood
