#!/usr/bin/env python3
""" trains a model for face verification using triplet loss"""

from triplet_loss import TripletLoss
import tensorflow.keras as K
import tensorflow as tf


class TrainModel:
    """ trains a model for face verification using triplet loss"""

    def __init__(self, model_path, alpha):
        """ Initialize Train Model
            - model_path is the path to the base face verification
            embedding model
            - loads the model using with
            tf.keras.utils.CustomObjectScope({'tf': tf}):
            - saves this model as the public instance method base_model
            - alpha is the alpha to use for the triplet loss calculation
            Creates a new model:
            inputs: [A, P, N]
            A is a numpy.ndarray containing the anchor images
            P is a numpy.ndarray containing the positive images
            N is a numpy.ndarray containing the negative images
            outputs: the triplet losses of base_model
            compiles the model with Adam optimization and no additional losses
            save this model as the public instance method training_model
            you can use from triplet_loss import TripletLoss
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = K.models.load_model(model_path)
        A_input = tf.keras.Input(shape=(96, 96, 3))
        P_input = tf.keras.Input(shape=(96, 96, 3))
        N_input = tf.keras.Input(shape=(96, 96, 3))

        predict_a = self.base_model(A_input)
        predict_b = self.base_model(P_input)
        predict_c = self.base_model(N_input)

        tl = TripletLoss(alpha)
        output = tl([predict_a, predict_b, predict_c])
        inputs = [A_input, P_input, N_input]

        training_model = K.models.Model(inputs, output)
        training_model.compile(optimizer='Adam')
        self.training_model = training_model

    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):
        """
        triplets is a list of numpy.ndarrayscontaining
            the inputs to self.training_model
        epochs is the number of epochs to train for
        batch_size is the batch size for training
        validation_split is the validation split for training
        verbose is a boolean that sets the verbosity mode
        """
        history = self.training_model.fit(x=triplets,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          validation_split=validation_split,
                                          verbose=verbose)
        return history

    def save(self, save_path):
        """
        save_path is the path to save the model
        eturns: the saved model
        """
        K.models.save_model(self.base_model, save_path)
        return self.base_model

    @staticmethod
    def f1_score(y_true, y_pred):
        """calcultes f1 score """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    @staticmethod
    def accuracy(y_true, y_pred):
        """ Calculates Metrics accuracy """
        return K.metrics.accuracy(y_true, y_pred)

    def best_tau(self, images, identities, thresholds):
        """ Best Tau """
