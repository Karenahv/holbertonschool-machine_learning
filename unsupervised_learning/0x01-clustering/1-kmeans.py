#!/usr/bin/env python3
"""
0. Initialize K-means
"""

import numpy as np


def update_clss(X, C):
    """ Update the clss array
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset that
            will be used for K-means clustering
            n is the number of data points
            d is the number of dimensions for each data point
        C is an array of centroids
    Returs: a np.array withe the updated classes
    """
    distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
    return np.argmin(distances, axis=0)


def initialize(X, k):
    """ initializes cluster centroids for K-means
            - X is a numpy.ndarray of shape (n, d) containing
                the dataset that will be used for K-means clustering
                - n is the number of data points
                - d is the number of dimensions for each data point
            - k is a positive integer containing the number of clusters
            The cluster centroids are initialized with a multivariate
                uniform distribution along each dimension in d:
                - The minimum values for the distribution should be the
                minimum values of X along each dimension in d
                - The maximum values for the distribution should be the
                maximum values of X along each dimension in d
            Returns: a numpy.ndarray of shape (k, d) containing the
                initialized centroids for each cluster, or None on failure
        """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None
    n, d = X.shape
    cluster_centroids = np.random.uniform(np.amin(X, axis=0),
                                          np.amax(X, axis=0), (k, d))
    return cluster_centroids


def kmeans(X, k, iterations=1000):
    """ Function that performs K-means on a dataset
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset
            n is the number of data points
            d is the number of dimensions for each data point
        k is a positive integer containing the number of clusters
        iterations is a positive integer containing the maximum number
            of iterations that should be performed
    Returns:
        C, clss, or None, None on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None
    try:
        C = initialize(X, k)

        for i in range(iterations):
            C_copy = C.copy()
            clss = update_clss(X, C)

            # Moving centroids
            for k in range(C.shape[0]):
                # Check if is [NaN]
                if (X[clss == k].size == 0):
                    C[k, :] = initialize(X, 1)
                else:
                    C[k, :] = (X[clss == k].mean(axis=0))
            if (C_copy == C).all():
                return C, clss

        clss = update_clss(X, C)

        return C, clss
    except Exception:
        return None, None
