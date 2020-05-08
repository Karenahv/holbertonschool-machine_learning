#!/usr/bin/env python3
"""deep Neural Network"""
import numpy as np


class DeepNeuralNetwork():
    """Defines a deep neural network"""

    def __init__(self, nx, layers):
        """ Class constructor.
        """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        layers_arr = np.array(layers)

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(len(layers)):
            if (type(layers[i]) is not int or layers[i] < 1):
                raise TypeError("layers must be a list of positive integers")
            key_W = "W{}".format(i + 1)
            key_b = "b{}".format(i + 1)
            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.weights[key_W] = w
            else:
                aux = np.sqrt(2 / layers[i - 1])
                w = np.random.randn(layers[i], layers[i - 1]) * aux
                self.weights[key_W] = w
            b = np.zeros((layers[i], 1))
            self.weights[key_b] = b
