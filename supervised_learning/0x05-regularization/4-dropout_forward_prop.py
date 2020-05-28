#!/usr/bin/env python3
"""
Forward Propagation with Dropout
"""
import numpy as np


def softmax(z):
    """takes as input a vector of K real numbers, and
    normalizes it into a probability distribution"""
    t = np.exp(z)
    a = np.exp(z) / np.sum(t, axis=0, keepdims=True)
    return a


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout
    """
    cache = {}
    cache["A0"] = X
    for lay in range(L):
        Al = cache["A" + str(lay)]
        Wl = weights["W" + str(lay + 1)]
        bl = weights["b" + str(lay + 1)]
        Zl = np.matmul(Wl, Al) + bl
        if lay != L - 1:
            a = np.sinh(Zl) / np.cosh(Zl)
            drop = np.random.binomial(1, keep_prob, (a.shape[0], a.shape[1]))
            cache["D" + str(lay + 1)] = drop
            a = np.multiply(a, drop)
            cache["A" + str(lay + 1)] = a/keep_prob
        else:
            cache["A" + str(lay + 1)] = softmax(Zl)
    return cache
