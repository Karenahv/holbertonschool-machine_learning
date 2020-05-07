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
        evaluation = np.where(output_neuron >= 0.5, 1, 0)
        error = self.cost(Y, output_neuron)
        return (evaluation, error)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Gradient descent"""
        m = len(X[0])
        dy_hat = A - Y
        dw = (1 / float(m)) * np.matmul(X, dy_hat.transpose())
        db = (1 / float(m)) * np.sum(dy_hat)
        self.__W = self.__W - (alpha * dw.transpose())
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ trains neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)