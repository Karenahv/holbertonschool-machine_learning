#!/usr/bin/env python3
""" Entropy and affinities  relative to a data point """

import numpy as np


def HP(Di, beta):
    # Apply pj|i equation
    Pi = np.exp(-Di * beta) / np.sum(np.exp(-Di * beta))
    # Apply shannon entropy equation
    Hi = - np.sum(Pi * np.log2(Pi))

    return Hi, Pi
