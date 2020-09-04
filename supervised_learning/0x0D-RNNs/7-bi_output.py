#!/usr/bin/env python3
"""class gru_cell"""

import numpy as np


class BidirectionalCell:
    """class gru_cell"""

    def __init__(self, i, h, o):
        """
        :param i:is the dimensionality of the data
        :param h:is the dimensionality of the hidden state
        :param o:is the dimensionality of the outputs

        Whf and bhfare for the hidden
         states in the forward direction
        Whb and bhbare for the hidden
         states in the backward direction
        Wy and byare for the outputs
        The weights should be initialized using
         a random normal distribution in the order listed above
        The weights will be used on the right side
         for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ calculates the hidden state in the forward direction
            for one time step
            - x_t is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
                - m is the batch size for the data
            - h_prev is a numpy.ndarray of shape (m, h) containing the
                previous hidden state
            Returns: h_next, the next hidden state
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        :param h_next: is a numpy.ndarray of shape
         (m, h) containing the next hidden state
        :param x_t: is a numpy.ndarray of shape (m, i)
         that contains the data input for the cell
        :return: h_pev, the previous hidden state
        """

        h_x = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(h_x, self.Whb) + self.bhb)

        return h_prev

    def softmax(self, x):
        """ softmax function """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def output(self, H):
        """ calculates all outputs for the RNN:
            - H is a numpy.ndarray of shape (t, m, 2 * h) that contains
                the concatenated hidden states from both directions,
                excluding their initialized states
                - t is the number of time steps
                - m is the batch size for the data
                - h is the dimensionality of the hidden states
            Returns: Y, the outputs
        """
        t = H.shape[0]
        Y = []
        for t_step in range(t):
            y = np.matmul(H[t_step], self.Wy) + self.by
            # you can use softmax instead softmax2 - same op
            y = self.softmax(y)
            Y.append(y)

        return np.array(Y)
