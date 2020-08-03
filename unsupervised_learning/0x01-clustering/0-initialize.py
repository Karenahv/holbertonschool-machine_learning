#!/usr/bin/env python3
"""Initialize K-means """

import numpy as np


def initialize(X, k):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None
    m = np.shape(X)[0]
    cluster_centroids = np.random.uniform(np.amin(X, axis=0),
                                          np.amax(X, axis=0), (k, m))
    return cluster_centroids
