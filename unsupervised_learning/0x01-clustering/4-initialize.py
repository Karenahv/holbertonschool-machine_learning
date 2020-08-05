#!/usr/bin/env python3
"""initializes variables for
    a Gaussian Mixture Model"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    You are not allowed to use any loops
    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing
    the priors for each cluster, initialized evenly
    m is a numpy.ndarray of shape (k, d) containing
    the centroid means for each cluster, initialized with K-means
    S is a numpy.ndarray of shape (k, d, d) containing
     the covariance matrices for each cluster, initialized as identity matrices
    You should use kmeans = __import__('1-kmeans').kmeans
    """
    C, clss = kmeans(X, k)
    n, d = X.shape
    pi = np.full((k,), 1 / k)
    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)
    return pi, C, S
