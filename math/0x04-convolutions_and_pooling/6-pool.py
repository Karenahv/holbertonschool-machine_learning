#!/usr/bin/env python3
""" Pooling images """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs pooling max or average:
    Returns: a numpy.ndarray containing the pooling images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    h_final = int(((h-kh)/sh) + 1)
    w_final = int(((h-kw)/sw) + 1)
    array = np.zeros((m, h_final, w_final, c))
    img = np.arange(m)
    for j in range(h_final):
        for i in range(w_final):
            if mode == 'max':
                array[img, j, i] = (np.max(images[img, j*sh:(kh+(j*sh)),
                                                  i*sw:(kw+(i*sw))],
                                           axis=(1, 2)))
            if mode == 'avg':
                array[img, j, i] = (np.mean(images[img, j*sh:(kh+(j*sh)),
                                                   i*sw:(kw+(i*sw))],
                                            axis=(1, 2)))
    return array
