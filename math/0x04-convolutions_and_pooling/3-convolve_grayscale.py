#!/usr/bin/env python3
""" Convolution """
import numpy as np


def convolve_grayscale(images, kernel, padding='same',
                       stride=(1, 1)):
    """ performs a convolution on grayscale images:
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]
    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]
    elif padding == 'same':
        ph = int(((h-1)*sh+kh-h)/2) + 1
        pw = int(((w-1)*sw+kw-w)/2) + 1
    else:
        ph = 0
        pw = 0
    if padding == 'same' or isinstance(padding, tuple):
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)
    h_final = int(((h+2*ph-kh)/sh) + 1)
    w_final = int(((w+2*pw-kw)/sw) + 1)
    array = np.zeros((m, h_final, w_final))
    img = np.arange(m)
    for j in range(h_final):
        for i in range(w_final):
            array[img, j, i] = (np.sum(images[img, j*sh:(kh+(j*sh)),
                                              i*sw:(kw+(i*sw))] *
                                       kernel, axis=(1, 2)))
    return array
