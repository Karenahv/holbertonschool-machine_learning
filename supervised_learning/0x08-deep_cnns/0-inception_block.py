#!/usr/bin/env python3
""" builds an inception block as described in
Going Deeper with Convolutions (2014)"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ builds an inception block as described in
Going Deeper with Convolutions (2014)"""
    kernel_init = K.initializers.he_normal(seed=None)
    conv_1x1 = K.layers.Conv2D(filters[0],
                               (1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=kernel_init,
                               )(A_prev)
    conv_1x1_b3 = K.layers.Conv2D(filters[1],
                                  (1, 1),
                                  padding='same',
                                  activation='relu',
                                  kernel_initializer=kernel_init,
                                  )(A_prev)
    conv_3x3 = K.layers.Conv2D(filters[2], (3, 3),
                               padding='same',
                               activation='relu',
                               kernel_initializer=kernel_init,
                               )(conv_1x1_b3)
    conv_1x1_b5 = K.layers.Conv2D(filters[3],
                                  (1, 1), padding='same',
                                  activation='relu',
                                  kernel_initializer=kernel_init,
                                  )(A_prev)
    conv_5x5 = K.layers.Conv2D(filters[4],
                               (5, 5),
                               padding='same',
                               activation='relu',
                               kernel_initializer=kernel_init,
                               )(conv_1x1_b5)

    pool_proj = K.layers.MaxPool2D((3, 3),
                                   strides=(1, 1),
                                   padding='same')(A_prev)
    conv_1x1_bpool = K.layers.Conv2D(filters[5],
                                     (1, 1),
                                     padding='same',
                                     activation='relu',
                                     kernel_initializer=kernel_init,
                                     )(pool_proj)

    output = K.layers.concatenate([conv_1x1,
                                   conv_3x3,
                                   conv_5x5,
                                   conv_1x1_bpool])
    return output
