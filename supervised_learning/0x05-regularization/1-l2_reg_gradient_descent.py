#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """GD with L2 regularization"""
    m = Y.shape[1]
    dz = 0
    temp_weights = weights.copy()

    for i in range(L, 0, -1):
        keyw = "W{}".format(i)
        keyb = "b{}".format(i)
        keyout = "A{}".format(i)
        X = "A{}".format(i - 1)

        if i == L:
            dz = (cache[keyout] - Y)
            dw = np.matmul(dz, cache[X].transpose()) / m
        else:
            d1 = 1 - (cache[keyout]) ** 2
            weight = temp_weights["W{}".format(i + 1)]
            dz_layer = np.matmul(weight.T, dz) * d1
            dw = (1/m) * np.matmul(dz_layer, cache[X].T)
            dz = dz_layer

        dw_reg = dw + (lambtha/m)*temp_weights[keyw]
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[keyw] = temp_weights[keyw] - (alpha * dw_reg)
        weights[keyb] = temp_weights[keyb] - (alpha * db)
