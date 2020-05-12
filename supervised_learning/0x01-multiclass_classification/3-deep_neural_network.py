#!/usr/bin/env python3
"""deep Neural Network"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


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
            if i == (self.__L - 1):
                t = np.exp(matmult_x_y + self.__weights["b" + str(index)])
                self.__cache[key_A] = t / np.sum(t, axis=0, keepdims=True)
            else:
                self.__cache[key_A] = 1 / (1 +
                                           np.exp(-matmult_x_y -
                                                  self.__weights["b" +
                                                                 str(index)]))
        return (self.__cache["A" + str(self.__L)], self.__cache)

    def cost(self, Y, A):
        """cost of the model using logistic regression"""
        lost = -(Y * np.log(A))
        cost_all = lost.sum()/len(Y[0])
        return cost_all

    def evaluate(self, X, Y):
        """neuron performing binary clasification"""
        self.forward_prop(X)
        long = self.__L
        max_val = np.amax(self.__cache["A" + str(long)], axis=0)
        evaluation = np.where(self.__cache["A" + str(long)] == max_val, 1, 0)
        error = self.cost(Y, self.__cache["A" + str(long)])
        return (evaluation, error)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the network """
        weights_copy = self.__weights.copy()

        m = Y.shape[1]

        A3 = self.__cache['A' + str(self.__L)]
        A2 = self.__cache['A' + str(self.__L - 1)]
        W3 = weights_copy['W' + str(self.__L)]
        b3 = weights_copy['b' + str(self.__L)]
        dz_List = {}
        dz3 = A3 - Y

        dz_List['dz'+str(self.__L)] = dz3
        dw3 = (1/m) * np.matmul(A2, dz3.T)
        db3 = (1/m) * np.sum(dz3, axis=1, keepdims=True)
        self.__weights['W'+str(self.__L)] = W3 - (alpha * dw3).T
        self.__weights['b'+str(self.__L)] = b3 - (alpha * db3)

        for i in range(self.__L - 1, 0, -1):
            A_curr = self.__cache['A'+str(i)]
            A_bef = self.__cache['A'+str(i - 1)]
            W_curr = weights_copy['W'+str(i)]
            W_next = weights_copy['W'+str(i + 1)]
            b_curr = weights_copy['b'+str(i)]
            dz1 = np.matmul(W_next.T, dz_List['dz'+str(i + 1)])
            dz2 = A_curr * (1 - A_curr)
            dz = dz1 * dz2
            dw = (1/m) * np.matmul(A_bef, dz.T)
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            dz_List['dz'+str(i)] = dz
            self.__weights['W'+str(i)] = W_curr - (alpha * dw).T
            self.__weights['b'+str(i)] = b_curr - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the neuron with more parameters """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        list_cost = []
        for ite in range(iterations):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            cost = self.cost(Y, A)
            list_cost.append(cost)
            if verbose:
                if (ite % step == 0):
                    print("Cost after {} iterations: {}".format(ite, cost))
        if graph:
            list_x = np.arange(0, iterations)
            list_y = list_cost
            plt.plot(list_x, list_y)
            plt.title('Training Cost')
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.show

        return self.evaluate(X, Y)

    def save(self, filename):
        """save the instance object to a file in pickle format"""
        if '.pkl' not in filename:
            filename = filename + '.pkl'
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)

    def load(filename):
        """ load file"""
        if os.path.exists(filename) is True:
            with open(filename, 'rb') as handle:
                open_file = pickle.load(handle)
            return open_file
        return None
