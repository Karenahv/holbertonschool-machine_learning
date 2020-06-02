#!/usr/bin/env python3
""" train a model using keras"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True,
                shuffle=False, learning_rate_decay=False,
                alpha=0.1, decay_rate=1,):
    """ train a model using keras"""
    def learning_rate(epoch):
        """ updates the learning rate using mini-batch gradient descent"""
        return alpha / (1 + decay_rate * epoch)
    callbacks = []
    if learning_rate_decay:
        decay = K.callbacks.LearningRateScheduler(learning_rate, 1)
        callbacks.append(decay)
    if early_stopping and validation_data:
        callback = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(callback)
    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          shuffle=shuffle, validation_data=validation_data,
                          callbacks=callbacks)
    return history
