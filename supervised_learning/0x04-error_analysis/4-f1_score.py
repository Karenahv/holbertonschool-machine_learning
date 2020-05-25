#!/usr/bin/env python3
"""F1 SCORE"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """returns F1 SCORE """

    sen = sensitivity(confusion)
    pre = precision(confusion)
    F1 = (2 * sen * pre) / (sen + pre)
    return F1
