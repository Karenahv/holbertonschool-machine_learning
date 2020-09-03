#!/usr/bin/env python3
"""class gru_cell"""

import numpy as np


class GRUCell:
    """class gru_cell"""

    def __init__(self, i, h, o):
        """
        :param i:is the dimensionality of the data
        :param h:is the dimensionality of the hidden state
        :param o:is the dimensionality of the outputs

        Wz and bz are for the update gate
        Wr and br are for the reset gate
        Wh and bh are for the intermediate hidden state
        Wy and by are for the output
        The weights should be initialized using
         a random normal distribution in the order listed above
        The weights will be used on the right side
         for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))


    def softmax(self, x):
        """ softmax function """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def sigmoid(self, x):
        """ sigmoid function """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """
        :param h_prev:  is a numpy.ndarray of shape (m, h)
         containing the previous hidden state
        :param x_t:x_t is a numpy.ndarray of shape (m, i)
         that contains the data input for the cell
        m is the batche size for the data
        :return:
        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell
        """

        matrix = np.concatenate((h_prev, x_t), axis=1)
        z_t = self.sigmoid(np.matmul(matrix, self.Wz) + self.bz)
        r_t = self.sigmoid(np.matmul(matrix, self.Wr) + self.br)

        matrix2 = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_proposal = np.tanh(np.matmul(matrix2, self.Wh) + self.bh)
        h_next = z_t * h_proposal + (1 - z_t) * h_prev

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
