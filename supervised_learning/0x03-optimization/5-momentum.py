#!/usr/bin/env python3
"""gradient descent with momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """gradient descent with momentum """
    Vdw = (beta1 * v) + ((1 - beta1) * grad)
    var = var - (alpha * Vdw)
    return (var, Vdw)
