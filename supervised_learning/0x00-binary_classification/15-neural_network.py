#!/usr/bin/env python3
"""Neural Network"""
import numpy as np


class NeuralNetwork():
    """Defines a neural network"""

    def __init__(self, nx, nodes):
        """ Class constructor..
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise TypeError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes * nx).reshape(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ W1 attribute getter.
        """
        return self.__W1

    @property
    def b1(self):
        """ b1 attribute getter.
        """
        return self.__b1

    @property
    def A1(self):
        """ A1 attribute getter.
        """
        return self.__A1

    @property
    def W2(self):
        """ W2 attribute getter.
        """
        return self.__W2

    @property
    def b2(self):
        """ b2 attribute getter
        """
        return self.__b2

    @property
    def A2(self):
        """ A2 attribute getter.
        """
        return self.__A2

    def forward_prop(self, X):
        """Forward Propagation"""
        matmult_x_y = np.matmul(self.__W1, X)
        self.__A1 = 1 / (1 + np.exp(-matmult_x_y - self.__b1))
        matmult_a1_w2 = np.matmul(self.__W2, self.__A1)
        self.__A2 = 1 / (1 + np.exp(-matmult_a1_w2 - self.__b2))
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """cost of the model using logistic regression"""
        lost = -(Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))
        cost_all = lost.sum()/len(Y[0])
        return cost_all

    def evaluate(self, X, Y):
        """neuron performing binary clasification"""
        self.__A1, self.__A2 = self.forward_prop(X)
        evaluation = np.where(self.__A2 >= 0.5, 1, 0)
        error = self.cost(Y, self.__A2)
        return (evaluation, error)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Gradient descent"""
        m = len(X[0])
        dy_hat = A1 - Y
        dy2_hat = A2 - Y
        temp = np.matmul(self.__W2.transpose(), dy2_hat)
        temp2 = (A1 * (1 - A1))
        dw = (1/float(m)) * (np.matmul(temp * temp2, X.transpose()))
        db = (1 / float(m)) * (temp * temp2).sum(axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dw)
        self.__b1 = self.__b1 - (alpha * db)
        dw2 = (1 / float(m)) * np.matmul(A1, dy2_hat.transpose())
        db2 = (1 / float(m)) * dy2_hat.sum(axis=1, keepdims=True)
        self.__W2 = self.__W2 - (alpha * dw2.transpose())
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ trains neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cont = 0
        cost_total = []
        eje_x = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if verbose:
                if i % step == 0 or step == iterations:
                    cost = self.cost(Y, self.__A2)
                    print("Cost after {} iterations: {}".format(i, cost))
                    cost_total.append(cost)
                    eje_x.append(i)
        if graph:
            plt.plot(eje_x, cost_total)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training cost')
            plt.show()
        return self.evaluate(X, Y)
