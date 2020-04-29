#!/usr/bin/env python3
""" class Exponential"""


e = 2.7182818285


class Exponential():
    """ Class to calculate Exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Constructor of Exponential
        """
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) > 1:
                    self.data = data
                    self.lambtha = float(1 / (sum(self.data) / len(self.data)))
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")

    def pdf(self, x):
        """ Calculates the Probability Density Function of the distribution.
        """
        if x < 0:
            return 0
        return self.lambtha * e**(-self.lambtha * x)

    def cdf(self, x):
        """ Calculates the cumulative distribution function
        """
        if x < 0:
            return 0

        cdf = 1 - e ** (-self.lambtha * x)
        return cdf
