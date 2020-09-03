#!/usr/bin/env python3
"""class gru_cell"""

import numpy as np


class LSTMCell:
    """class gru_cell"""

    def __init__(self, i, h, o):
        """
        :param i:is the dimensionality of the data
        :param h:is the dimensionality of the hidden state
        :param o:is the dimensionality of the outputs

        Wfand bf are for the forget gate
        Wuand bu are for the update gate
        Wcand bc are for the intermediate cell state
        Woand bo are for the output gate
        Wyand by are for the outputs
        The weights should be initialized using
         a random normal distribution in the order listed above
        The weights will be used on the right side
         for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """ softmax function """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def sigmoid(self, x):
        """ sigmoid function """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev,  x_t):
        """
        :param h_prev:  is a numpy.ndarray of shape (m, h)
         containing the previous hidden state
        :param x_t:x_t is a numpy.ndarray of shape (m, i)
         that contains the data input for the cell
        m is the batche size for the data
        :param c_prev is a numpy.ndarray of shape (m, h)
         containing the previous cell state
        Returns: h_next, c_next, y
        h_next is the next hidden state
        c_next is the next cell state
        y is the output of the cell
        """

        matrix = np.concatenate((h_prev, x_t), axis=1)
        u_t = self.sigmoid(np.matmul(matrix, self.Wu) + self.bu)
        f_t = self.sigmoid(np.matmul(matrix, self.Wf) + self.bf)
        o_t = self.sigmoid(np.matmul(matrix, self.Wo) + self.bo)
        c_tilde = np.tanh(np.matmul(matrix, self.Wc) + self.bc)
        c_next = u_t * c_tilde + f_t * c_prev
        h_next = o_t * np.tanh(c_next)

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
