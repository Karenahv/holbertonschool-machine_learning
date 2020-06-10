#!/usr/bin/env python3
"""LeNet-5 architecture using tensorflow"""


import tensorflow as tf


def lenet5(x, y):
    """LeNet-5 architecture using tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer()
    # Step 1: Convolution
    layer_convo1 = tf.layers.Conv2D(filters=6, kernel_size=5,
                                    activation='relu',
                                    padding="same",
                                    kernel_initializer=init)(x)
    # Step 2: Max Pooling Layer
    layer_poolmax = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                 strides=(2, 2))(layer_convo1)
    # step 3: Convolution
    layer_convo2 = tf.layers.Conv2D(filters=16, kernel_size=5,
                                    activation='relu',
                                    padding="valid",
                                    kernel_initializer=init)(layer_poolmax)
    # step 4: Max pooling 2
    layer_poolmax2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                  strides=(2, 2))(layer_convo2)

    layer_flat = tf.layers.Flatten()(layer_poolmax2)

    # step 5: Fully connected layer 120 nodes
    layer_fc = tf.layers.Dense(units=120,
                               activation='relu',
                               kernel_initializer=init)(layer_flat)
    # step 6: Fully connected layer 84 nodes
    layer_fc2 = tf.layers.Dense(units=84,
                                activation='relu',
                                kernel_initializer=init)(layer_fc)

    # step 7: Fully connected softmax output layer with 10 nodes
    output = tf.layers.Dense(units=10,
                             kernel_initializer=init)(layer_fc2)

    # Calculate loss
    loss = tf.losses.softmax_cross_entropy(y, output)

    #  tensor for the softmax activated output
    tensor_softmax = tf.nn.softmax(output)

    # Training operation that utilizes Adam optimization
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy of the network
    values = tf.math.argmax(y, 1)
    prediction = tf.math.argmax(output, 1)
    compare = tf.math.equal(values, prediction)
    accuracy = tf.math.reduce_mean(tf.cast(compare, tf.float32))

    return (tensor_softmax, optimazer, loss, accuracy)
