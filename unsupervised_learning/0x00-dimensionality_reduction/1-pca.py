#!/usr/bin/env python3
"""Performs PCA on dataset"""

import numpy as np


def pca(X, ndim):
    """Performs PCA on dataset"""
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:ndim])
    return Y
