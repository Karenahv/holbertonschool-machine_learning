# !/usr/bin/env python3
"""
5. Q affinities
"""

import numpy as np


def Q_affinities(Y):
    """ Function that calculates the Q affinities
    Args:
        Y is a numpy.ndarray of shape (n, ndim) containing the low
            dimensional transformation of X
    Returns: Q, num
        Q - is a numpy.ndarray of shape (n, n) containing the Q affinities
        num - is a numpy.ndarray of shape (n, n) containing the numerator
            of the Q affinities
    Documentation:
        https://lvdmaaten.github.io/tsne/
    """
    n, ndim = Y.shape
    # (a-b)^2 = a^2 + b^2 - 2*a*b
    Y_square = np.sum(np.square(Y), axis=1)
    XY = np.dot(Y, Y.T)
    D = np.add(np.add((-2 * XY), Y_square).T, Y_square)

    Q = np.zeros((n, n))
    num = np.zeros((n, n))
    for i in range(n):
        Di = D[i].copy()
        Di = np.delete(Di, i, axis=0)
        numerator = (1 + Di) ** (-1)
        numerator = np.insert(numerator, i, 0)
        num[i] = numerator

    den = num.sum()
    Q = num / den
    return Q, num
