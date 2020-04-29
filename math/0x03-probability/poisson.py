#!/usr/bin/env python3
""" class poisson"""

# Euler
e = 2.7182818285


class Poisson():
    """ class Poisson"""
    def __init__(self, data=None, lambtha=1.):
        """constructor"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if isinstance(data, list):
                if (len(data) > 1):
                    self.data = data
                    self.lambtha = sum(self.data)/len(self.data)
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")

    def pmf(self, k):
        """ calculates pmf"""
        k = int(k)
        if k < 0:
            return (0)
        dnominador = 1
        for i in range(1, k + 1):
            dnominador = dnominador * i
        pmf = (e ** -self.lambtha * self.lambtha ** k)/dnominador
        return pmf

    def cdf(self, k):
        """cdf"""
        if k < 0:
            return (0)
        k = int(k)
        cdf = 0
        for i in range(k + 1):
            cdf = cdf + self.pmf(i)
        return cdf
