#!/usr/bin/env python3
"""performs convolution on grayscale images"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """""performs convolution on grayscale images"""
    m = len(images)
    h = len(images[0])
    w = len(images[1])
    hk = len(kernel)
    wk = len(kernel[0])
    h_final = (h - hk) + 1
    w_final = (w - wk) + 1
    sh = stride[0]
    sw = stride[1]
    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]
    elif (padding == "same"):
        ph = int(((h-1)*sh+hk-h)/2) + 1
        pw = int(((w-1)*sw+wk-w)/2) + 1
    else:
        ph = 0
        pw = 0
    if padding == 'same' or isinstance(padding, tuple):
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)
    h_final = int(((h+2*ph-hk)/sh) + 1)
    w_final = int(((w+2*pw-wk)/sw) + 1)
    array = np.zeros((m, h_final, w_final))
    img = np.arange(m)
    for i in range(int(h_final)):
        for j in range(int(w_final)):
            array[img, i, j] = (np.sum(images[img, i * sh:(hk + (i * sh)),
                                              j * sw:(wk + (j * sw))] *
                                       kernel, axis=(1, 2)))
    return array
