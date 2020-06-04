#!/usr/bin/env python3
""" Same Convolution """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ performs a same convolution on grayscale images:
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = padding[0]
    pw = padding[1]
    new_img = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                     mode='constant', constant_values=0)
    h_new = new_img.shape[1]
    w_new = new_img.shape[2]
    h_final = (h_new - kh) + 1
    w_final = (w_new - kw) + 1
    array = np.zeros((m, h_final, w_final))
    img = np.arange(m)
    for j in range(h_final):
        for i in range(w_final):
            array[img, j, i] = (np.sum(new_img[img, j:kh+j, i:kw+i] *
                                       kernel, axis=(1, 2)))
    return array
