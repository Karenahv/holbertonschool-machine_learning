#!/usr/bin/env python3
""" builds a transition layer as described
 in Densely Connected Convolutional
Networks"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ builds a transition layer as described
    in Densely Connected Convolutional
    Networks Which do convolution and pooling.
    Works as downsampling."""

    kernel_init = K.initializers.he_normal(seed=None)
    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Convolution2D(int(nb_filters * compression),
                               (1, 1), padding='same',
                               kernel_initializer=kernel_init)(x)

    x = K.layers.AveragePooling2D((2, 2), strides=None,
                                  padding='same')(x)
    return x, int(nb_filters * compression)
