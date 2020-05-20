#!/usr/bin/env python3
""" updates a variable using Adam Optimization"""


def update_variables_Adam(alpha, beta1,
                          beta2,
                          epsilon, var,
                          grad, v, s, t):
    """updates a variable using Adam Optimization"""
    Vdw = (beta1 * v) + ((1 - beta1) * grad)
    Sdw = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    Vdw_corrected = Vdw / (1 - (beta1 ** t))
    Sdw_corrected = Sdw / (1 - (beta2 ** t))
    var = var - (alpha * (Vdw_corrected / ((Sdw_corrected ** (1/2)) +
                                           epsilon)))
    return var, Vdw, Sdw
