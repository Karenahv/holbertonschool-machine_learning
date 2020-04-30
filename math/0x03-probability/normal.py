#!/usr/bin/env python3
""" class normal"""


e = 2.7182818285
pi = 3.1415926536


class Normal():
    """ Class to calculate Normal distribution"""

    def __init__(self, data=None, mean=0, stddev=1.):
        """Constructor of Normal
        """
        self.stddev = float(stddev)
        self.mean = float(mean)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) > 1:
                    self.data = data
                    self.mean = float(sum(self.data) / len(self.data))
                    numerador = 0
                    for item in data:
                        numerador = numerador + ((item - self.mean) ** 2)
                    self.stddev = (numerador / len(data)) ** 0.5
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")

    def z_score(self, x):
        """ Calculates z score.
      """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ calculates x_value
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """ Calculates pdf distribution function
        """
        exponente = (((x - self.mean) / self.stddev) ** 2) * -0.5
        denominador = self.stddev * ((2 * pi) ** 0.5)
        pdf = (1 / denominador) * (e ** exponente)
        return pdf

    def cdf(self, x):
        """ Calculates cdf distribution function
        """
        aux = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = (2 / (pi ** 0.5)) * \
              (aux - (aux ** 3) / 3 + (aux ** 5) / 10 - (aux ** 7) / 42 +
               (aux ** 9) / 216)
        cdf = (0.5) * (1 + erf)
        return cdf
