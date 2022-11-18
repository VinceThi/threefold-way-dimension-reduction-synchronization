# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


import numpy as np
from numpy.linalg import eigh

""" Important note
Use scipy.sparse.linalg.eigs or scipy.sparse.linalg.eigsh if you don't need 
the complete spectrum.
"""


def get_eigenvectors_matrix(adjacency_matrix, n):
    """
    We obtain the n dominant right eigenvectors of a symmetric
    adjacency matrix A = A^T. In the spectral dimension reduction, we need
    the left eigenvectors, but when A = A^T, the right eigenvectors are equal
    to the transpose of the left eigenvectors.

    WARNING: This function is not efficient because we don't need to compute
    all the eigenvalues and eigenvectors. The functions
          scipy.sparse.linalg.eigs or scipy.sparse.linalg.eigsh
    should be used instead.

       :param adjacency_matrix: N by N real symmetric matrix
       :param n: number of dominant eigenvectors  n < N
       :return: eigenvector_matrix: non normalized dominant eigenvectors matrix
                                    shape=(n, N)
                                    Each row is a dominant eigenvector and the
                                    row 0 is a more or equally dominant
                                    eigenvector to row 1, row 1 is a more or
                                    equally dominant eigenvector to row 2, etc.
    """
    eigenvector_matrix = np.zeros((n, len(adjacency_matrix[:, 0])))
    eigenvalues, eigenvectors = eigh(adjacency_matrix)

    i = 0
    lower_index = 0
    upper_index = -1
    while i < n:
        if np.absolute(np.absolute(eigenvalues[upper_index])
                       - np.absolute(eigenvalues[lower_index])) < 0.01:
            if i == 0:  # Then, it is the dominant eigenvector.
                eigenvector_matrix[i, :] = \
                    np.absolute(eigenvectors[:, upper_index])
            else:
                eigenvector_matrix[i, :] = \
                    eigenvectors[:, upper_index]
            upper_index -= 1
            i += 1
        elif np.absolute(eigenvalues[upper_index]) \
                > np.absolute(eigenvalues[lower_index]):
            if i == 0:  # Then, it is the dominant eigenvector.
                eigenvector_matrix[i, :] = \
                    np.absolute(eigenvectors[:, upper_index])
            else:
                eigenvector_matrix[i, :] = \
                    eigenvectors[:, upper_index]
            upper_index -= 1
            i += 1
        else:
            if i == 0:  # Then, it is the dominant eigenvector.
                eigenvector_matrix[i, :] = \
                    np.absolute(eigenvectors[:, lower_index])
            else:
                eigenvector_matrix[i, :] = eigenvectors[:, lower_index]
            lower_index += 1
            i += 1
    return eigenvector_matrix


def get_laplacian_eigenvectors_matrix(laplacian_matrix):
    """

       :param laplacian_matrix: n by n real symmetric matrix
       :return: eigenvector_matrix: non normalized dominant eigenvectors matrix
                                    shape=(2,n)
    """
    eigenvector_matrix = np.zeros((2, len(laplacian_matrix[:, 0])))
    eigenvalues, eigenvectors = eigh(laplacian_matrix)

    eigenvector_matrix[0, :] = np.absolute(eigenvectors[:, 1])
    eigenvector_matrix[1, :] = np.absolute(eigenvectors[:, 2])
    return eigenvector_matrix
