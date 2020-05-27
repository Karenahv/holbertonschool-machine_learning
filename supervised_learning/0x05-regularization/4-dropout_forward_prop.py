#!/usr/bin/env python3
"""forward propagation using dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """forward propagation using dropout"""
    cache = {}
    cache["A0"] = X
    for i in range(1, L + 1):
        keyw = "W{}".format(i)
        keyb = "b{}".format(i)
        keyA = "A{}".format(i - 1)

        z = np.matmul(weights[keyw], cache[keyA])
        z = z + weights[keyb]
        if i != L:
            drop = np.random.binomial(1, keep_prob, size=z.shape) / keep_prob
            A = drop * np.tanh(z)
            cache["D{}".format(i)] = drop
        else:
            t = np.exp(z)
            A = t / np.sum(t, axis=0, keepdims=True)

        cache["A" + str(i)] = A
    return cache
