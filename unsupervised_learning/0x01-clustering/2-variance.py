#!/usr/bin/env python3
""" Variance """

import numpy as np


def variance(X, C):
    """ Function that calculates the total intra-cluster variance for a
        data set
    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing the centroid
            means for each cluster
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(C, np.ndarray) or len(C.shape) != 2:
            return None
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        distancia_min = np.min(distances, axis=0)
        var = np.sum(distancia_min ** 2)
        return var
    except Exception:
        return None
