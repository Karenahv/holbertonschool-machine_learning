#!/usr/bin/env pytho3
""" RMSProp optimization algorithm"""


def update_variables_RMSProp(alpha,
                             beta2, epsilon,
                             var, grad, s):
    """Update a variable using RMSProp"""
    Stw = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var = var - ((alpha * grad) / ((Stw ** (1/2)) + epsilon))
    return var, Stw
