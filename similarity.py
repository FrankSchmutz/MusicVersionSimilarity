import numpy as np


def signature_error(a, b):
    """
    Computes the error between two distributions
    of dominant notes

    Keyword arguments:
    a -- the first distribution of dominant notes
    b -- the second distribution of dominant notes

    Return:
    the error between the two distributions
    """

    # We return half to total absolute error to get a result between 0 and 1
    return np.sum(np.abs(a - b)) / 2


def signatures_similarity(a, b):
    """
    Compute the similarity between two signed tracks

    Keyword arguments:
    a -- the first signed track
    b -- the second signed track

    Return:
    the similarity between the two signed tracks
    """

    errors = np.array([signature_error(i, j) for i, j in zip(a, b)])
    return np.round(1.0 - np.mean(errors), 2)
