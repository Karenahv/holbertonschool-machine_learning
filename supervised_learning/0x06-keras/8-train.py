#!/usr/bin/env python3
""" train a model using keras"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True,
                shuffle=False):
    """ train a model using keras"""
    def learning_rate(epoch):
        """ updates the learning rate using mini-batch gradient descent"""
        return alpha / (1 + decay_rate * epoch)
    callbacks = []
    if save_best:
        checkpoint = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        callbacks.append(checkpoint)
    if learning_rate_decay:
        decay = K.callbacks.LearningRateScheduler(learning_rate, 1)
        callbacks.append(decay)
    if early_stopping and validation_data:
        callback = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(callback)
    history = network.fit(data, labels, epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          verbose=verbose,
                          callbacks=callbacks)
    return history
