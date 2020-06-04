#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """""performs a valid convolution on grayscale images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[1]
    hk = kernel.shape[0]
    wk = kernel.shape[1]
    h_final = (h - hk) + 1
    w_final = (w - wk) + 1
    array = np.zeros((m, h_final, w_final))
    img = np.arange(m)
    for i in range(h_final):
        for j in range(w_final):
            array[img, i, j] = (np.sum(images[img, i:hk + i, j:wk + j] *
                                       kernel, axis=(1, 2)))
    return array
