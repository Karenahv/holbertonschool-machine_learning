#!/usr/bin/env python3
""" Initialize """
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal Distribution"""
    def __init__(self, data):
        """
        - data is a numpy.ndarray of shape (d, n) containing the data set:
                - n is the number of data points
                - d is the number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("X must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("X must contain multiple data points")
        n = data.shape[1]
        d = data.shape[0]
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        deviation = np.tile(self.mean.reshape(-1), n).reshape(n, d)
        cov = data.T - deviation
        self.cov = np.matmul(cov.T, cov) / (n - 1)
