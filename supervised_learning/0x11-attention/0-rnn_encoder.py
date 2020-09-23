#!usr/bin/env/python3
"""class rnnEncoder"""

import tensorflow as tf


class RNNEncoder:
    """Class rnnencoder"""

    def __init__(self, vocab, embedding, units, batch):
        """
        :param vocab: is an integer representing
         the size of the input vocabulary
        :param embedding: is an integer representing
         the dimensionality of the embedding vector
        :param units: is an integer representing the
        number of hidden units in the RNN cell
        :param batch: is an integer representing
         the batch size
        """

        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        :return:a tensor of shape (batch, units)containing
         the initialized hidden states
        """
        initializer = tf.keras.initializers.Zeros()
        values = initializer(shape=(self.batch, self.units))
        return values

    def call(self, x, initial):
        """
        :param x:  is a tensor of shape (batch, input_seq_len)
         containing the input to the encoder layer as word
        indices within the vocabulary
        :param initial: is a tensor of shape (batch, units)
         containing the initial hidden state
        :return: outputs, hidden
        outputs is a tensor of shape (batch, input_seq_len,
         units)containing the outputs of the encoder
        hidden is a tensor of shape (batch, units)
         containing the last hidden state of the encoder
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
