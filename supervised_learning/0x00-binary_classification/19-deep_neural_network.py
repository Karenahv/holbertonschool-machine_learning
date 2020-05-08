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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

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

    @property
    def cache(self):
        """
        getter cache attribute info
        """
        return self.__cache

    @property
    def L(self):
        """
        getter L attribute info
        """
        return self.__L

    @property
    def weights(self):
        """
        getter weights attribute info
        """
        return self.__weights

    def forward_prop(self, X):
        """Forward Propagation"""
        self.__cache["A0"] = X
        for i in range(self.__L):
            key_A = "A{}".format(i)
            index = i + 1
            matmult_x_y = np.matmul(self.__weights["W" + str(index)],
                                    self.__cache[key_A])
            key_A = "A{}".format(i + 1)
            self.__cache[key_A] = 1 / (1 +
                                       np.exp(-matmult_x_y -
                                              self.__weights["b" +
                                                             str(index)]))
        return (self.__cache["A" + str(self.__L)], self.__cache)

    def cost(self, Y, A):
        """cost of the model using logistic regression"""
        lost = -(Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))
        cost_all = lost.sum()/len(Y[0])
        return cost_all
