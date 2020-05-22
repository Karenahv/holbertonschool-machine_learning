#!/usr/bin/env python3
"""
RMSprop Optimization Algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Returns: the updated variable and the new moment, respectively
    """
    Sd = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var = var - ((alpha * grad) / ((Sd ** (1/2)) + epsilon))
    return (var, Sd)
