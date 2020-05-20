#!/usr/bin/env python3
"""normalizes an unactivated output of
 a neural network using batch normalization """

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes an unactivated output"""
    mean = np.mean(Z, axis=0)
    varianza = np.var(Z, axis=0)
    Z_norm = (Z - mean) / ((varianza + epsilon)**(1/2))
    Z_new = gamma*Z_norm + beta
    return Z_new
