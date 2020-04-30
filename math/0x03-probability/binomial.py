#!/usr/bin/env python3
""" class Exponential"""


e = 2.7182818285


def factorial(n):
    """ return n!"""
    if n == 0:
        return 1
    total = 1
    for i in range(1, n + 1):
        total = total * i
    return total


class Binomial():
    """ Class to calculate Exponential distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Constructor of Binomial
        """
        self.p = float(p)
        self.n = int(n)
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if isinstance(data, list):
                if len(data) > 1:
                    self.data = data
                    mean = float(sum(self.data) / len(self.data))
                    numerador = 0
                    for item in data:
                        numerador = numerador + ((item - mean) ** 2)
                    stddev = (numerador / len(data)) ** 0.5
                    variance = stddev ** 2
                    p = 1 - variance / mean
                    self.n = int(round(mean / p))
                    self.p = mean / self.n
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")

    def pmf(self, k):
        """ Calculates the Probability Density Function of the distribution.
        """
        if k < 0:
            return 0
        k = int(k)
        c = factorial(int(self.n)) / (factorial(k)
                                      * factorial(int(self.n) - k))
        other = (self.p ** k) * ((1 - self.p) ** (int(self.n) - k))
        pmf = c * other
        return pmf

    def cdf(self, k):
        """ Calculates the cumulative distribution function
        """
        if k < 0:
            return 0
        k = int(k)

        cdf = 0
        for i in range(k + 1):
            cdf = cdf + self.pmf(i)
        return cdf
