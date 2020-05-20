#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""

import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    vt = 0
    mov_avg = []
    for i in range(len(data)):
        vt = beta * vt + (1 - beta) * data[i]
        mov_avg.append(vt / (1 - beta ** (i + 1)))
    return mov_avg
