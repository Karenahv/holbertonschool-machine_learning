# !/usr/bin/env python3
""" Cost """
import numpy as np


def cost(P, Q):
    """ calculates the cost of the t-SNE transformation
    - P is a numpy.ndarray of shape (n, n) containing the P affinities
    - Q is a numpy.ndarray of shape (n, n) containing the Q affinities
    Returns: C, the cost of the transformation
    """
    Q = np.where(Q != 0, Q, 1e-12)
    P = np.where(P != 0, P, 1e-12)
    cost = np.sum(P * np.log(P / Q))
    return cost
