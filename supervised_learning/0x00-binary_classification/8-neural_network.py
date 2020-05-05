#!/usr/bin/env python3
""" single neuron performing
    binary classification
"""

import numpy as np


class NeuralNetwork():
    """ class neural Network"""
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nx < 1:
            raise ValueError("nodes must be a positive integer")
        w1 = [np.random.randn(nx).tolist() for i in range(nodes)]
        self.W1 = np.array(w1)
        self.b1 = [np.zeros(1).tolist() for i in range(nodes)]
        self.A1 = 0
        w2 = np.random.randn(nodes)
        self.W2 = np.array([w2])
        self.b2 = 0
        self.A2 = 0
