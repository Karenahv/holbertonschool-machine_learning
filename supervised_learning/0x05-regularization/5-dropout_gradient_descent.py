#!/usr/bin/env python3
""" Gradient Descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """GD with dropout regularization"""
    temp_weights = weights.copy()
    dz = 0
    m = Y.shape[1]
    for i in range(L, 0, -1):
        keyw = "W{}".format(i)
        keyb = "b{}".format(i)
        keyout = "A{}".format(i)
        X = "A{}".format(i - 1)

        if i == L:
            dz = (cache[keyout] - Y)
            dw = np.matmul(dz, cache[X].T) / m
        else:
            d1 = 1 - ((cache[keyout]) * (cache[keyout]))
            weight = temp_weights["W{}".format(i + 1)]
            drop = (cache["D{}".format(i)] / keep_prob) * d1
            dz_l = np.matmul(weight.T, dz) * drop
            dw = (np.matmul(dz_l, cache[X].T)) / m
            dz = dz_l
        db = (dz.sum(axis=1, keepdims=True)) / m
        weights[keyw] = temp_weights[keyw] - (alpha * (dw))
        weights[keyb] = temp_weights[keyb] - (alpha * db)
