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
        self.__W1 = np.array(w1)
        self.__b1 = [np.zeros(1).tolist() for i in range(nodes)]
        self.__A1 = 0
        w2 = np.random.randn(nodes)
        self.__W2 = np.array([w2])
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        return Weights first layer
        """
        return self.__W1

    @property
    def W2(self):
        """
        return Weights hidden layer
        """
        return self.__W2

    @property
    def b1(self):
        """
        return bias first layer
        """
        return self.__b1

    @property
    def b2(self):
        """
        return bias hidden layer
        """
        return self.__b2

    @property
    def A1(self):
        """
        return Active hidden layer 1
        """
        return self.__A1

    @property
    def A2(self):
        """
        return Active output 2
        """
        return self.__A2
