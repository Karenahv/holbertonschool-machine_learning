#!/usr/bin/env python3
"""shuffle data"""

import numpy as np


def shuffle_data(X, Y):
    """shiffle data"""
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    return X_shuffled, Y_shuffled
