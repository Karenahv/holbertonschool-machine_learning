#!/usr/bin/env python3
"""Hello, sklearn!"""

import sklearn.mixture


def gmm(X, k):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset
        k is the number of clusters
    Returns: pi, m, S, clss, bic
    pi is a numpy.ndarray of shape (k,) containing the cluster priors
    m is a numpy.ndarray of shape (k, d)
    containing the centroid means
    S is a numpy.ndarray of shape (k, d, d)
    containing the covariance matrices
    clss is a numpy.ndarray of shape (n,) containing
    the cluster indices for each data point
    bic is a numpy.ndarray of shape (kmax - kmin + 1)
    containing the BIC value for each cluster size tested
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    m = gmm.means_
    S = gmm.covariances_
    pi = gmm.weights_
    labels = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, labels, bic
