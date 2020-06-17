#!/usr/bin/env python3
""" that builds the ResNet-50 architecture
as described in Deep Residual Learning for
Image Recognition (2015)"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ that builds the ResNet-50 architecture
    as described in Deep Residual Learning for
    Image Recognition (2015)"""

    kernel_init = K.initializers.he_normal(seed=None)
    A_prev = K.Input(shape=(224, 224, 3))
    x = K.layers.Conv2D(64,
                        (7, 7),
                        padding='same',
                        strides=2,
                        kernel_initializer=kernel_init)(A_prev)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPool2D(pool_size=3,
                           strides=2,
                           padding='same')(x)
    # Conv2_x
    # 1×1, 64
    # 3×3, 64
    # 1×1, 256

    x = projection_block(x, [64, 64, 256], 1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    # Conv3_x
    # 1×1, 128
    # 3×3, 128
    # 1×1, 512

    x = projection_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    # Conv4_x
    # 1×1, 256
    # 3×3, 256
    # 1×1, 1024

    x = projection_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    # 1×1, 512
    # 3×3, 512
    # 1×1, 2048

    x = projection_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    # average pool, 1000-d fc, softmax

    x = K.layers.AveragePooling2D(pool_size=7,
                                  strides=None,
                                  padding='same')(x)

    x = K.layers.Dense(1000,
                       activation='softmax',
                       kernel_regularizer=K.regularizers.l2())(x)
    model = K.models.Model(A_prev, x)

    return model
