#!/usr/bin/env python3
"""performs a same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """""performs a same convolution on grayscale images"""
    m = len(images)
    h = len(images[0])
    w = len(images[1])
    hk = len(kernel)
    wk = len(kernel[0])
    h_final = (h - hk) + 1
    w_final = (w - wk) + 1
    ph = int((hk - 1) / 2)
    pw = int((wk - 1) / 2)
    if hk % 2 == 0:
        ph = int(kh/2)
    if wk % 2 == 0:
        pw = int(kw/2)
    new_img = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                     mode='constant', constant_values=0)
    h_final = h_final + 2 * ph
    w_final = w_final + 2 * pw
    array = np.zeros((m, h_final, w_final))
    img = np.arange(m)
    for i in range(int(h_final)):
        for j in range(int(w_final)):
            array[img, i, j] = (np.sum(new_img[img, i:hk + i, j:wk + j] *
                                       kernel, axis=(1, 2)))
    return array
