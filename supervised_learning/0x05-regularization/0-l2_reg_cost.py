#!/usr/bin/env python3
"""
L2 Regularization Cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization
    """
    frobenius = 0
    for layer in range(1, L + 1):
        key = 'W' + str(layer)
        frobenius += np.linalg.norm(weights[key])
    L2 = cost + ((lambtha/(2*m)) * frobenius)
    return L2
