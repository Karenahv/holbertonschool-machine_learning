#!/usr/bin/env python3
""" single neuron performing
    binary classification
"""

import numpy as np


class Neuron():
    """ class neuron bainary
    classification"""
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        w = np.random.randn(nx)
        self.__W = np.array([w])
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter weights"""
        return self.__W

    @property
    def b(self):
        """getter private bias"""
        return self.__b

    @property
    def A(self):
        """getter private output neuron A"""
        return self.__A

    def forward_prop(self, X):
        """Forward Propagation"""
        matmult_x_y = np.matmul(self.__W, X)
        self.__A = 1 / (1 + np.exp(-matmult_x_y - self.__b))
        return self.__A

    def cost(self, Y, A):
        """cost of the model using logistic regression"""
        lost = -(Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))
        cost_all = lost.sum()/len(Y[0])
        return cost_all

    def evaluate(self, X, Y):
        """neuron performing binary clasification"""
        output_neuron = self.forward_prop(X)
        evaluation = np.array([1 if i >= 0.5 else 0 for i in output_neuron[0]])
        error = self.cost(Y, output_neuron)
        return (evaluation, error)
