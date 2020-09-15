#!/usr/bin/env python3
"""bag of words"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    :param sentences: list of sentences to analize
    :param vocab:is a list of the
    vocabulary words to use for the analysis
    :return: embeddings, features
    embeddings is a numpy.ndarray
     of shape (s, f) containing the embeddings
    s is the number of sentences in sentences
    f is the number of features analyzed
    features is a list of the features used for embeddings
    """

    input_vectorizer = CountVectorizer(vocabulary=vocab)
    output_vectorizer = input_vectorizer.fit_transform(sentences)
    return output_vectorizer.toarray(), input_vectorizer.get_feature_names()
