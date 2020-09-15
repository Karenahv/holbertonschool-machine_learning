#!/usr/bin/env python3
"""tfid"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
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

    tfidf = TfidfVectorizer(vocabulary=vocab)
    output = tfidf.fit_transform(sentences)
    tfidf.get_feature_names()
    return output.toarray(), tfidf.get_feature_names()
