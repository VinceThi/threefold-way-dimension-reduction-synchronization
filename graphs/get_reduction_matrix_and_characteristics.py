# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

from numpy.linalg import pinv, det, eig
import numpy as np
from synch_predictions.graphs.matrix_factorization_pymf_snmf import *
from synch_predictions.graphs.graph_spectrum import get_eigenvectors_matrix
import json
import tkinter.simpledialog
from tkinter import messagebox
import time
import networkx as nx
from tqdm import tqdm
# Note: I was not able to jit from numba for snmf and onmf to get reduction mat

# -------------------------- Matrix conditions --------------------------------


def matrix_is_singular(C):
    boolean = 0
    if np.abs(det(C)) < 1e-8:
        boolean = 1
    return boolean


def matrix_is_negative(M):
    return np.any(M < -1e-8)


def matrix_has_rank_n(M):
    n = len(M[:, 0])
    return np.linalg.matrix_rank(M) == n


def matrix_is_normalized(M):
    if len(np.shape(M)) == 1:
        bool_value = np.absolute(np.sum(M) - 1) < 0.000001
    else:
        n = len(M[:, 0])
        bool_value = \
            np.all(np.absolute(np.sum(M, axis=1) - np.ones(n)) < 0.000001)
    return bool_value


def matrix_is_orthogonal(M):
    boolean = 0
    X = np.abs(M@M.T - np.identity(np.shape(M)[0]))
    Y = np.abs(X - np.diag(np.diag(X)))
    # Because we don't want to check normalization, we substract the diagonal
    # of X to X to get Y. If it is a zero matrix, then M is orthogonal.
    if np.all(Y < 1e-8):
        boolean = 1
    return boolean


def matrix_is_orthonormalized_VV_T(V):
    n = len(V[:, 0])
    return np.all(np.absolute(V@V.T - np.identity(n)) < 1e-8)


def matrix_is_positive(M):
    return np.all(M >= -1e-8)


# ----------------- Algorithms to get the reduced matrices --------------------

def get_second_target_coefficent_matrix(V_T1, V_T2, V_T3,
                                        other_procedure=True):
    """
    This function is useful in a three-target procedure.
    :param V_T1: First target eigenvector matrix
    :param V_T2: Second target eigenvector matrix or a null matrix
                 np.zeros((n, N)) if it is not a two target procedure
    :param V_T3: Third target eigenvector matrix or a null matrix
                 np.zeros((n, N)) if it is not a three target procedure

    :param other_procedure: If True, it is the procedure 4 in the sync article
                            If False, it is the procedure 3 in the sync article
                            The name of the procedure will probably change
                            in the article,

    :return:
    """
    n, N = np.shape(V_T1)

    if np.all(V_T3 < 1e-10):
        C_T2 = np.zeros((n, n))
        # See the documentation of the function
        # get_first_target_coefficent_matrix(C_T2, V_T1, V_T2)

    else:
        if other_procedure:
            C_T2 = V_T3@pinv(V_T1)@V_T1@pinv(V_T2)
        else:
            C_T2 = V_T3@pinv(V_T2)

    return C_T2


def get_first_target_coefficent_matrix(C_T2, V_T1, V_T2):
    """
    :param V_T1: First target eigenvector matrix
    :param V_T2: Second target eigenvector matrix
    :param C_T2: Second_target_coefficent_matrix
                 without the normalization (C_T3 would be the normalization
                 matrix)

                 Can either be

                 1 - np.zeros((n, n)) for a two-target procedure
                 which only means that C_T2 will be use
                 to respect condition B (normalization and non negativity)

                 2 - a square matrix obtained with
                 get_second_target_coefficent_matrix
                 for a three-target procedure

    :return: C_T1:Coefficent matrix for target 1
                  without the normalization (it is related to C_T2 in a three
                  target procedure. In a two target procedure, C_T2 is related
                  to the normalization)
    """
    n = len(V_T1[:, 0])
    if np.all(C_T2 == np.zeros((n, n))):
        C_T1 = V_T2@pinv(V_T1)
    else:
        C_T1 = C_T2@V_T2@pinv(V_T1)
    return C_T1


def snmf(M, niter=500, W_init=None, H_init=None):
    """
    SNMF: Semi-nonnegative matrix factorization
    :param M: n x N matrix (n > N)
    :param niter: number of iteration in the algorithm, 100 iterations is a
                  safe number of iterations, see Ding 2010
    :param H_init:
    :param W_init:
    :return:
    """
    n, N = np.shape(M)
    snmf_mdl = SNMF(M, H_init=H_init, W_init=W_init, num_bases=n)
    snmf_mdl.factorize(niter=niter)
    # ---------------------------- Normalized frobenius error
    return snmf_mdl.W, snmf_mdl.H, snmf_mdl.ferr[-1]/(n*N)**2


def snmf_multiple_inits(M, number_initializations):
    """
    Notation: W -> F   and   H -> G
    :param M:
    :param number_initializations:
    :return:
    """
    n, N = np.shape(M)

    """ ---------------------------- SVD ---------------------------------- """
    u, s, vh = np.linalg.svd(M)

    # Initial matrix H with SVD
    G_init = np.absolute(vh[0:n, :])

    # Semi nonnegative matrix factorization
    # with SVD initialization
    F_svd, G_svd, frobenius_error_svd = snmf(M, H_init=G_init)
    # if not matrix_is_singular(F_svd):
    F, G = F_svd, G_svd
    snmf_frobenius_error = frobenius_error_svd
    print("snmf_frobenius_error_svd = ", snmf_frobenius_error)
    # if matrix_is_singular(F):
    #     for j in range(number_initializations):
    #         # Semi nonnegative matrix factorization
    #         # with random initialization
    #         F_random, G_random, frobenius_error_random = snmf(M, H_init=None)
    #         print(det(F_random))
    #         if not matrix_is_singular(F_random):
    #             F, G, = F_random, G_random
    #             snmf_frobenius_error = frobenius_error_random
    #
    # else:

    """ -------------------------- Random --------------------------------- """
    for j in range(number_initializations):
        # Semi nonnegative matrix factorization
        # with random initialization
        F_random, G_random, frobenius_error_random = snmf(M, H_init=None)
        # print(det(F_random))
        if snmf_frobenius_error > frobenius_error_random:
            F, G, = F_random, G_random
            snmf_frobenius_error = frobenius_error_random
            print("snmf_frobenius_error_random = ", snmf_frobenius_error)

    # print("snmf_frobenius_error_svd = ", snmf_frobenius_error)

    if matrix_is_singular(F):
        raise ValueError("W is singular in the semi-nonnegative matrix"
                         " factorization (snmf).")
    # ---------- Normalized frobenius error
    return F, G, snmf_frobenius_error


def onmf(M, max_iter=500, W_init=None, H_init=None):
    """
    Orthogonal Non-negative Matrix Factorization of X as X =WH wit HH^T=I.
    Based on Ref. Wang, Y. X., & Zhang, Y. J. (2012).
    Nonnegative matrix factorization: A comprehensive review.
    IEEE Transactions on Knowledge and Data Engineering, 25(6), 1336-1353.

    and

    https://github.com/mstrazar/iONMF

    ----------
    Input
    ----------
    M: array [n x N]
        Data matrix to be factorized.
    max_iter: int
        Maximum number of iterations.
    H_init: array [n x n]
        Fixed initial basis matrix.
    W_init: array [n x N]
        Fixed initial coefficient matrix.
    MoreOrtho: Boolean
        If True, searches for a matrix H with more zeros
    ---------
    Output
    ---------
    W: array [n x n]
    H: array [n x N]
    error: ||X-WH||/(nN)^2
        normalized factorization error
    o_error:  ||I-HH^T||/(n^2)
        normalized orthogonality error

    ex: SVD initialization
    n,N = X.shape
    # SVD
    u,s,vh = np.linalg.svd(X)
    # Initial matrix H
    h_init= abs(vh[0:n,:])
    # NMF
    W,H,e,oe = onmf(X, H_init = h_init)

    """

    n, N = np.shape(M)

    # add a small value, otherwise nmf and related methods get
    # into trouble as they have difficulties recovering from zero.
    W = np.random.random((n, n)) + 10**(-4) if isinstance(W_init, type(None))\
        else W_init
    H = np.random.random((n, N)) + 10**(-4) if isinstance(H_init, type(None))\
        else H_init

    for itr in range(max_iter):
        # update H
        numerator = W.T@M
        denominator = H@M.T@W@H
        H = np.nan_to_num(H*numerator/denominator)

        # new lines added to get orthonormalized rows
        row_norm = np.sqrt(np.diag(H@H.T))
        normalization_matrix = np.linalg.inv(np.diag(row_norm))
        H = normalization_matrix@H

        # update W
        numerator = M@H.T
        denominator = W@H@H.T
        W = np.nan_to_num(W*numerator/denominator)

    # error with normalized Frobenius norm
    error = np.linalg.norm(M - W@H)/(n*N)**2

    # orthogonality error with Frobenius norm
    o_error = np.linalg.norm(np.eye(n, n) - H@H.T)/(n**2)

    # ---------- Normalized frobenius error and normalized orthogonal error
    return W, H, error, o_error


def onmf_multiple_inits(M, number_initializations):
    """
    Notation: W -> F   and   H -> G
    :param M:
    :param number_initializations:
    :return:
    """
    n, N = np.shape(M)

    """ ---------------------------- SVD ---------------------------------- """
    u, s, vh = np.linalg.svd(M)

    # Initial matrix H with SVD
    G_init = np.absolute(vh[0:n, :])

    # Ortogonal nonnegative matrix factorization
    # with SVD initialization
    F_svd, G_svd, frobenius_error_svd, ortho_error_svd \
        = onmf(M, H_init=G_init)
    F, G = F_svd, G_svd
    onmf_frobenius_error, onmf_ortho_error = \
        frobenius_error_svd, ortho_error_svd

    print(f"\nonmf_frobenius_error_svd = {onmf_frobenius_error}",
          f"\nonmf_ortho_error_svd = {onmf_ortho_error}")

    """ --------------------------- Random -------------------------------- """

    for j in range(number_initializations):
        # Ortogonal nonnegative matrix factorization
        # with random initialization
        F_random, G_random, frobenius_error_random, ortho_error_random \
            = onmf(M, H_init=None)   # S'assurer que c'est ok

        # 1. The condition below is the one used for transitions vs. n in the
        #    reply to the referee. The errors are normalized.
        # if frobenius_error_random**2 + ortho_error_random**2 < \
        #         onmf_frobenius_error**2 + onmf_ortho_error**2:

        # 2. We can penalize the orthogonal errors by adding weights if we want
        # if 0.1*frobenius_error_random**2 + 0.9*ortho_error_random**2 < \
        #         0.1*onmf_frobenius_error**2 + 0.9*onmf_ortho_error**2:
        #

        # 3. The condition below is the one used for FIG. 6 and 7 of the paper
        # if frobenius_error_random < onmf_frobenius_error:

        if frobenius_error_random**2 + ortho_error_random**2 < \
                onmf_frobenius_error**2 + onmf_ortho_error**2:
            F, G, = F_random, G_random
            onmf_frobenius_error, onmf_ortho_error = \
                frobenius_error_random, ortho_error_random
            # print("Result improved by a random initialization !")
            print(f"onmf_frobenius_error_random = {onmf_frobenius_error}",
                  f"\nonmf_ortho_error_random = {onmf_ortho_error}")

    if matrix_is_singular(F):
        ValueError('F is singular.')

    # print("onmf_frobenius_error_final = ", onmf_frobenius_error,
    #       "\nonmf_ortho_error_final = ", onmf_ortho_error)

    # ---------- Normalized frobenius error  and normalized ortho errors
    return F, G, onmf_frobenius_error, onmf_ortho_error


def normalize_rows_matrix_M1(M):
    return (M.T / np.sum(M, axis=1)).T


def normalize_rows_complex_matrix_M1(M):
    return (M.T / np.sum(M, axis=1)).T


def normalize_rows_matrix_VV_T(V):
    return (V.T / np.sqrt(np.sum(V**2, axis=1))).T


def get_reduction_matrix(V_T1, V_T2, V_T3, number_initializations=2000,
                         other_procedure=True):
    """
    Get the reduction matrix M for the dimension-reduction.

    :param V_T1: First target eigenvector matrix
    :param V_T2: Second target eigenvector matrix or a null matrix
                 np.zeros((n, N)) if it is not a two target procedure
    :param V_T3: Third target eigenvector matrix or a null matrix
                 np.zeros((n, N)) if it is not a three target procedure
    :param number_initializations:
            Number of different initializations of the semi and the
            orthogonal nonnegative matrix factorization (SNMF and ONMF).
            If niter=1, the algorithm will initialize SNMF and ONMF with SVD.
            If niter>1, the algorithm will initialize SNMF and ONMF with SVD in
            the first iteration and then, it will try random
            initializations to find the lowest Frobenius norm error
            frobenius_error = ||M - WH||.
    :param other_procedure: If True, it is the procedure 4 in the sync article
                            If False, it is the procedure 3 in the sync article
                            The name of the procedure will probably change
                            in the article,

    :return: M : n x N positive array/matrix. np.sum(M[mu,:], axis=1) = 1
                 for all mu which means that the matrix is normalized according
                 to its rows (the sum over the columns is one for each row)
    """

    n, N = np.shape(V_T1)
    # print(f"\nV_T1 = {V_T1}, \n V_T2 = {V_T2}, \n V_T3 = {V_T3}")

    if not np.all(V_T2 == np.zeros((n, N))):
        # Then it is a two or three targets procedure

        op = other_procedure
        C_T2 = get_second_target_coefficent_matrix(V_T1, V_T2, V_T3,
                                                   other_procedure=op)
        C_T1 = get_first_target_coefficent_matrix(C_T2, V_T1, V_T2)

        V = C_T1 @ V_T1

        # print(np.linalg.norm(np.eye(n, n) - V@V.T)/(n**2))
        # print("V_T1 = ", V_T1)
        # print("V_T2 = ", V_T2)
        # print("V = ", V)
        # print(f"\n C_T1 = {C_T1}, \n C_T2 = {C_T2}")
        # print(f"\ndet(C_T1) = {det(C_T1)}, \n det(C_T2) = {det(C_T2)}")

        if matrix_is_negative(V):
            F_snmf, G_snmf, snmf_frobenius_error = \
                snmf_multiple_inits(V, number_initializations)
            # print(V_T1, "\n", Q, "\n", U, det(Q))
            # print(f"Q = {Q}, \ndet(Q) = {det(Q)}")
            M_possibly_not_ortho = G_snmf
            # import matplotlib.pyplot as plt
            print("\nsnmf_ferr = ", snmf_frobenius_error)
            # plt.matshow(M_possibly_not_ortho, aspect="auto")
            # plt.colorbar()
            # plt.show()
        else:
            M_possibly_not_ortho = V
            # F_snmf = None
            snmf_frobenius_error = None

    else:
        # Then it is a one target procedure
        if matrix_is_negative(V_T1):
            F_snmf, G_snmf, snmf_frobenius_error = \
                snmf_multiple_inits(V_T1, number_initializations)
            # print(V_T1, "\n", Q, "\n", U, det(Q))
            # print(f"Q = {Q}, \ndet(Q) = {det(Q)}")
            # print(np.linalg.norm(V_T1 - F_snmf@G_snmf))
            print("\nsnmf_ferr = ", snmf_frobenius_error)
            M_possibly_not_ortho = G_snmf
        else:
            M_possibly_not_ortho = V_T1
            # F_snmf = None
            snmf_frobenius_error = None

    # if matrix_is_negative(M_possibly_not_ortho):
    #     raise ValueError('The function '
    #                      'get_non_normalized_positive_reduction_matrix'
    #                      'does not reach its goal.'
    #                      ' There is probably an error in the function.')
    # if matrix_is_negative(M_possibly_not_ortho):
    if not matrix_is_orthogonal(M_possibly_not_ortho):
        # If the matrix is not already orthogonal...
        F_onmf, M_not_normalized, onmf_frobenius_error, onmf_ortho_error =\
            onmf_multiple_inits(M_possibly_not_ortho,
                                number_initializations)
        # import matplotlib.pyplot as plt
        print(f"\nonmf_ferr = {onmf_frobenius_error} ",
              f"\nonmf_oerr = {onmf_ortho_error}")
        # plt.matshow(M_possibly_not_ortho, aspect="auto")
        # plt.colorbar()
        # plt.show()
        # if not np.all(V_T2 == np.zeros((n, N))):
        #     if matrix_is_negative(V):
        #         print("||V - F_snmf@F_onmf@M_not_normalized|| = ",
        #               np.linalg.norm(V - F_snmf@F_onmf@M_not_normalized))
        # else:
        #     if matrix_is_negative(V_T1):
        #         print("||V - F_snmf@F_onmf@M_not_normalized|| = ",
        #               np.linalg.norm(V_T1
        #                              - F_snmf@F_onmf@M_not_normalized))

    else:
        M_not_normalized = M_possibly_not_ortho
        onmf_frobenius_error, onmf_ortho_error = None, None

    M = normalize_rows_matrix_M1(M_not_normalized)

    if not matrix_is_positive(M):
        raise ValueError("The reduced matrix M is not positive anymore after"
                         "using orthonormal matrix factorization.")

    return M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error


def get_CVM_dictionary(W, K, A, V_W, V_K, V_A, graph_str, parameter_dictionary,
                       other_procedure=True):
    """
    Get a dictionary containing all the reduction matrices for a given set of
    parameters in the diagonal matrix W, a given graph of adjacency matrix A
    and diagonal matrix of degrees K with a choice of eigenvector matrices
    related to W, K, A.

    :param W: Frequency matrix
    :param K: Degree matrix
    :param A: Adjacency matrix
    :param V_W: Eigenvector matrix of the frequency matrix
    :param V_K: Eigenvector matrix of the degree matrix
    :param V_A: Eigenvector matrix of the adjacency matrix
    :param graph_str: A string that indicate the type of graph
    :param parameter_dictionary: All the parameter we want to save
                                 in the CVM dictionary. Note that the keys in
                                 this dictionary must be different than the one
                                 in the CVM_dictionary
    :param other_procedure: If True, it is the procedure 4 in the sync article
                    If False, it is the procedure 3 in the sync article
                    The name of the procedure will probably change
                    in the article,
    :return:
    """

    n = len(V_W[:, 0])
    CVM_dictionary = {"W": W.tolist(), "K": K.tolist(), "A": A.tolist()}
    CVM_dictionary.update(parameter_dictionary)
    V_none = np.zeros(np.shape(V_W))
    # --------------------------- One target ----------------------------------

    """ W """
    T1, T2, T3 = "W", "None", "None"
    print(T1+"->"+T2+"->"+T3+"\n")
    V_T1, V_T2, V_T3 = V_W, V_none, V_none
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)

    properties_W = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, "None", "None",
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_W"] = V_T1.tolist()
    CVM_dictionary["M_W"] = M.tolist()

    """ K """
    T1, T2, T3 = "K", "None", "None"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_K, V_none, V_none
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_K = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, "None", "None",
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_K"] = V_T1.tolist()
    CVM_dictionary["M_K"] = M.tolist()

    """ A """
    T1, T2, T3 = "A", "None", "None"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_A, V_none, V_none
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_A = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, "None", "None",
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_A"] = V_T1.tolist()
    CVM_dictionary["M_A"] = M.tolist()

    # ---------------------------- Two target ---------------------------------

    """ W -> K """
    T1, T2, T3 = "W", "K", "None"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_W, V_K, V_none
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_WK = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, "None",
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_WK"] = V_T1.tolist()
    CVM_dictionary["V_T2_WK"] = V_T2.tolist()
    CVM_dictionary["M_WK"] = M.tolist()

    """ W -> A """
    T1, T2, T3 = "W", "A", "None"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_W, V_A, V_none
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_WA = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, "None",
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_WA"] = V_T1.tolist()
    CVM_dictionary["V_T2_WA"] = V_T2.tolist()
    CVM_dictionary["M_WA"] = M.tolist()

    """ K -> W """
    T1, T2, T3 = "K", "W", "None"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_K, V_W, V_none
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_KW = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, "None",
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_KW"] = V_T1.tolist()
    CVM_dictionary["V_T2_KW"] = V_T2.tolist()
    CVM_dictionary["M_KW"] = M.tolist()

    """ K -> A """
    T1, T2, T3 = "K", "A", "None"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_K, V_A, V_none
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_KA = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, "None",
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_KA"] = V_T1.tolist()
    CVM_dictionary["V_T2_KA"] = V_T2.tolist()
    CVM_dictionary["M_KA"] = M.tolist()

    """ A -> W """
    T1, T2, T3 = "A", "W", "None"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_A, V_W, V_none
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_AW = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, "None",
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_AW"] = V_T1.tolist()
    CVM_dictionary["V_T2_AW"] = V_T2.tolist()
    CVM_dictionary["M_AW"] = M.tolist()

    """ A -> K """
    T1, T2, T3 = "A", "K", "None"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_A, V_K, V_none
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_AK = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, "None",
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_AK"] = V_T1.tolist()
    CVM_dictionary["V_T2_AK"] = V_T2.tolist()
    CVM_dictionary["M_AK"] = M.tolist()

    # --------------------------- Three target --------------------------------
    """ W -> K -> A """
    T1, T2, T3 = "W", "K", "A"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_W, V_K, V_A
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_WKA = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, V_T3,
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_WKA"] = V_T1.tolist()
    CVM_dictionary["V_T2_WKA"] = V_T2.tolist()
    CVM_dictionary["V_T3_WKA"] = V_T3.tolist()
    CVM_dictionary["M_WKA"] = M.tolist()

    """ W -> A -> K """
    T1, T2, T3 = "W", "A", "K"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_W, V_A, V_K
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_WAK = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, V_T3,
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_WAK"] = V_T1.tolist()
    CVM_dictionary["V_T2_WAK"] = V_T2.tolist()
    CVM_dictionary["V_T3_WAK"] = V_T3.tolist()
    CVM_dictionary["M_WAK"] = M.tolist()

    """ K -> W -> A """
    T1, T2, T3 = "K", "W", "A"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_K, V_W, V_A
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_KWA = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, V_T3,
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_KWA"] = V_T1.tolist()
    CVM_dictionary["V_T2_KWA"] = V_T2.tolist()
    CVM_dictionary["V_T3_KWA"] = V_T3.tolist()
    CVM_dictionary["M_KWA"] = M.tolist()

    """ K -> A -> W """
    T1, T2, T3 = "K", "A", "W"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_K, V_A, V_W
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_KAW = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, V_T3,
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_KAW"] = V_T1.tolist()
    CVM_dictionary["V_T2_KAW"] = V_T2.tolist()
    CVM_dictionary["V_T3_KAW"] = V_T3.tolist()
    CVM_dictionary["M_KAW"] = M.tolist()

    """ A -> W -> K """
    T1, T2, T3 = "A", "W", "K"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_A, V_W, V_K
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_AWK = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, V_T3,
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_AWK"] = V_T1.tolist()
    CVM_dictionary["V_T2_AWK"] = V_T2.tolist()
    CVM_dictionary["V_T3_AWK"] = V_T3.tolist()
    CVM_dictionary["M_AWK"] = M.tolist()

    """ A -> K -> W """
    T1, T2, T3 = "A", "K", "W"
    print(T1 + "->" + T2 + "->" + T3 + "\n")
    V_T1, V_T2, V_T3 = V_A, V_K, V_W
    M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
        get_reduction_matrix(V_T1, V_T2, V_T3, other_procedure=other_procedure)
    properties_AKW = \
        get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                        V_T1, V_T2, V_T3,
                                                        M,
                                                        snmf_frobenius_error,
                                                        onmf_frobenius_error,
                                                        onmf_ortho_error,
                                                        W, K, A)
    CVM_dictionary["V_T1_AKW"] = V_T1.tolist()
    CVM_dictionary["V_T2_AKW"] = V_T2.tolist()
    CVM_dictionary["V_T3_AKW"] = V_T3.tolist()
    CVM_dictionary["M_AKW"] = M.tolist()

    properties_list = \
        ["One target", properties_W, properties_K, properties_A,
         "Two target", properties_WK, properties_WA, properties_KW,
         properties_KA, properties_AW, properties_AK,
         "Three target", properties_WKA, properties_WAK, properties_KWA,
         properties_KAW, properties_AWK, properties_AKW]

    if messagebox.askyesno("Python",
                           "Would you like to save the dictionary: "
                           "CVM_dictionary?"):
        window = tkinter.Tk()
        window.withdraw()  # hides the window
        file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

        timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

        f = open(f'CVM_data/{timestr}_CVM_{graph_str}_{n}D_properties_{file}'
                 '.txt', 'w')
        f.writelines(properties_list)

        f.close()

        with open(f'CVM_data/{timestr}_CVM_dictionary_{graph_str}_{n}D_{file}'
                  f'.json', 'w') as outfile:
            json.dump(CVM_dictionary, outfile)


def get_CVM_dictionary_absolute_path(dictionary_name_string):

    if dictionary_name_string == "bipartite_winfree_kuramoto":
        CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
                        "synch_predictions/graphs/SBM/CVM_data/" \
                        "2020_02_13_11h59min42sec_CVM_dictionary_bipartite" \
                        "_2D_pout_0_2_omega1_0_3.json"

    elif dictionary_name_string == "SBM_winfree_kuramoto":
        CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
                        "synch_predictions/graphs/SBM/CVM_data/" \
                        "2020_02_13_12h29min13sec_CVM_dictionary_SBM" \
                        "_2D_p11_0_7_p22_0_5_pout_0_2_omega1_0_3.json"

    elif dictionary_name_string == "bipartite_theta":
        CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
                        "synch_predictions/graphs/SBM/CVM_data/" \
                        "2020_02_13_12h58min55sec_CVM_dictionary_bipartite" \
                        "_2D_pout_0_2_omega1_m1_1_omega2_m0_9.json"

    elif dictionary_name_string == "SBM_theta":
        CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
                        "synch_predictions/graphs/SBM/CVM_data/" \
                        "2020_02_13_12h49min55sec_CVM_dictionary_SBM" \
                        "_2D_p11_0_7_p22_0_5_pout_0_2" \
                        "_omega1_m1_1_omega2_m0_9.json"

    else:
        raise ValueError("The given dictionary_name_string is not valid.")

    return CVM_dict_path


def get_random_graph_realizations(graph_generator, nb_realizations, file,
                                  *args):
    for i in tqdm(range(nb_realizations)):
        A = nx.to_numpy_array(graph_generator(*args)).tolist()
        with open(f'SBM_instances/{file}/'
                  f'A{i}.json', 'w') as outfile:
            json.dump(A, outfile)
    return


def get_M_realizations(V_T1, V_T2, V_T3, nb_realizations):
    M_list = []
    for _ in tqdm(range(nb_realizations)):
        M = get_reduction_matrix(V_T1, V_T2, V_T3, number_initializations=100,
                                 other_procedure=True)[0]
        M_list.append(M)
    return np.array(M_list)


def get_omega_realizations(nb_realizations, mean, std, N1, N2,
                           dynamics_str="kuramoto"):

    omega_realizations = []
    for _ in tqdm(range(nb_realizations)):
        if dynamics_str == "theta":
            current_mismatch = 0.2
            omega1_random = -np.random.normal(mean,
                                              std, N1)
            omega2_random = -np.random.normal(mean - current_mismatch,
                                              std, N2)
            omega = np.concatenate([omega1_random, omega2_random])
            omega_realizations.append(omega.tolist())
        else:
            omega1_random = np.random.normal(mean, std, N1)
            omega2_random = -N1 / N2 * omega1_random[:N2]
            # print(omega1_random, "\n", omega2_random)
            # This choice of omega2 does not make the sum of
            # omega1_random + omega2_random is equal to zero
            error_sum_omega1_omega2 = \
                np.sum(np.concatenate([omega1_random, omega2_random]))
            term_to_be_added_to_omega2_random = error_sum_omega1_omega2 / N2
            omega2_random -= term_to_be_added_to_omega2_random * np.ones(N2)
            omega = np.concatenate([omega1_random, omega2_random])
            omega_realizations.append(omega.tolist())
            if np.sum(omega) > 1e-5:
                raise ValueError("The natural frequencies don't add up to 0.")
    return omega_realizations


def get_realizations_dictionary(omega_realizations, N1, N2,
                                nb_realizations, graph_generator,
                                args_graph_generator):

    adjacency_matrix_realizations = []
    M_realizations = []
    snmf_ferr_realizations = []
    onmf_ferr_realizations = []
    onmf_oerr_realizations = []

    for i in tqdm(range(nb_realizations)):
        while True:
            try:
                A = nx.to_numpy_array(graph_generator(*args_graph_generator))
                adjacency_matrix_realizations.append(A.tolist())
                N = len(A[:0])
                omega = np.array(omega_realizations[i])
                omega1 = omega[:N1]
                omega2 = omega[N1:]
                omega1_norm = np.abs(omega1 / np.sqrt(np.sum(omega1**2)))
                omega2_norm = np.abs(omega2 / np.sqrt(np.sum(omega2**2)))
                V_W = np.block([[omega1_norm, np.zeros(N2)],
                                [np.zeros(N1), omega2_norm]])
                V_A = get_eigenvectors_matrix(A, 2)

                M, snmf_frobenius_error, onmf_frobenius_error,\
                    onmf_ortho_error,\
                    = get_reduction_matrix(V_A, V_W, np.zeros((2, N)),
                                           number_initializations=100,
                                           other_procedure=True)
                if M[0][0] < M[0][-1]:  # Warning, in principle, its ok for
                                        # dense graph TODO
                    M[[0, 1]] = M[[1, 0]]
                M_realizations.append(M.tolist())
                snmf_ferr_realizations.append(snmf_frobenius_error)
                onmf_ferr_realizations.append(onmf_frobenius_error)
                onmf_oerr_realizations.append(onmf_ortho_error)
                break
            except np.linalg.LinAlgError:
                print("A singular matrix was obtain while looking for a "
                      "reduction matrix M")

    realizations_dictionary = {"adjacency_matrix_realizations":
                               adjacency_matrix_realizations,
                               "M_realizations": M_realizations,
                               "snmf_ferr_realizations":
                                   snmf_ferr_realizations,
                               "onmf_ferr_realizations":
                                   onmf_ferr_realizations,
                               "onmf_oerr_realizations":
                                   onmf_oerr_realizations,
                               "T_1": "A", "T_2": "W", "T_3": "None"}

    return realizations_dictionary


def get_realizations_dictionary_absolute_path(dictionary_name_string):

    if dictionary_name_string == "bipartite_winfree":
        realizations_dictionary = \
            "2020_02_19_20h39min45sec_realizations_dictionary" \
            "_for_winfree_on_bipartite_2D_parameter_realizations_True"

        """ Target A, N = 250"""
        # "2020_02_16_23h19min21sec_realizations_dictionary" \
        # "_for_winfree_on_bipartite_2D_parameter_realizations_False"

        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_15_18h01min36sec_realizations" \
        # "_dictionary_for_winfree_on_bipartite_2D"
        """ N = 500 """
        # "2020_02_16_15h21min11sec_realizations" \
        # "_dictionary_for_winfree_on_bipartite_2D"

    elif dictionary_name_string == "SBM_winfree":
        realizations_dictionary = \
            "2020_02_19_21h02min17sec_realizations_dictionary" \
            "_for_winfree_on_SBM_2D_parameter_realizations_True"

        " Target A, N = 250"
        # "2020_02_16_23h28min50sec_realizations_dictionary" \
        # "_for_winfree_on_SBM_2D_parameter_realizations_False"

        # (below) With different parameters realizations
        """ N = 250 """
        # realizations_dictionary ="2020_02_15_18h33min11sec_realizations" \
        #                          "_dictionary_for_winfree_on_SBM_2D"
        """ N = 500 """
        # "2020_02_16_15h56min03sec_realizations" \
        # "_dictionary_for_winfree_on_SBM_2D"

    elif dictionary_name_string == "bipartite_kuramoto":
        realizations_dictionary = \
            "2020_02_19_20h39min44sec_realizations_dictionary" \
            "_for_kuramoto_on_bipartite_2D_parameter_realizations_True"

        """ Target A, N = 250 """
        # "2020_02_16_23h22min31sec_realizations_dictionary" \
        # "_for_kuramoto_on_bipartite_2D_parameter_realizations_False"

        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_15_18h00min03sec_realizations" \
        # "_dictionary_for_kuramoto_on_bipartite_2D"
        """ N = 500 """
        # "2020_02_16_15h23min15sec_realizations" \
        # "_dictionary_for_kuramoto_on_bipartite_2D"

    elif dictionary_name_string == "SBM_kuramoto":
        realizations_dictionary = \
            "2020_02_19_21h03min15sec_realizations_dictionary" \
            "_for_kuramoto_on_SBM_2D_parameter_realizations_True"

        """ Target A, N = 250 """
        # "2020_02_16_23h28min49sec_realizations_dictionary" \
        # "_for_kuramoto_on_SBM_2D_parameter_realizations_False"
        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_15_18h33min25sec_realizations" \
        # "_dictionary_for_kuramoto_on_SBM_2D"
        """ N = 500 """
        # "2020_02_16_15h56min44sec_realizations" \
        # "_dictionary_for_kuramoto_on_SBM_2D"

    elif dictionary_name_string == "bipartite_theta":
        realizations_dictionary = \
            "2020_02_19_20h39min46sec_realizations_dictionary" \
            "_for_theta_on_bipartite_2D_parameter_realizations_True"

        """ Target A, N = 250"""
        # "2020_02_16_23h24min21sec_realizations_dictionary" \
        # "_for_theta_on_bipartite_2D_parameter_realizations_False"

        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_16_22h14min08sec_realizations" \
        # "_dictionary_for_theta_on_bipartite_2D"
        """ N = 500 """
        # "2020_02_16_15h22min34sec_realizations" \
        # "_dictionary_for_theta_on_bipartite_2D"

    elif dictionary_name_string == "SBM_theta":
        realizations_dictionary = \
            "2020_02_19_21h02min36sec_realizations_dictionary" \
            "_for_theta_on_SBM_2D_parameter_realizations_True"

        """ Target A, N = 250"""
        # "2020_02_16_23h29min26sec_realizations_dictionary" \
        # "_for_theta_on_SBM_2D_parameter_realizations_False"

        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_16_22h19min24sec_realizations" \
        # "_dictionary_for_theta_on_SBM_2D"
        """ N = 500 """
        # "2020_02_16_15h55min49sec_realizations" \
        # "_dictionary_for_theta_on_SBM_2D"

    else:
        raise ValueError("The given dictionary_name_string is not valid.")

    return realizations_dictionary


def get_infos_realizations_dictionary_absolute_path(dictionary_name_string):

    if dictionary_name_string == "bipartite_winfree":
        realizations_dictionary_str = \
            "2020_02_19_20h39min45sec_infos_realizations_dictionary" \
            "_for_winfree_on_bipartite_2D_parameter_realizations_True"

        """ Target A, N = 250"""
        # "2020_02_16_23h19min21sec_infos_realizations" \
        # "_dictionary_for_winfree_on_bipartite_2D" \
        # "_parameter_realizations_False"

        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_15_18h01min36sec_infos_realizations" \
        # "_dictionary_for_winfree_on_bipartite_2D"
        """ N = 500 """
        # "2020_02_16_15h21min11sec_infos_realizations" \
        # "_dictionary_for_winfree_on_bipartite_2D"

    elif dictionary_name_string == "SBM_winfree":
        realizations_dictionary_str = \
            "2020_02_19_21h02min17sec_infos_realizations_dictionary" \
            "_for_winfree_on_SBM_2D_parameter_realizations_True"

        """ Target A, N = 250 """
        # "2020_02_16_23h28min50sec_infos_realizations" \
        # "_dictionary_for_winfree_on_SBM_2D" \
        # "_parameter_realizations_False"
        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_15_18h33min11sec_infos_realizations" \
        # "_dictionary_for_winfree_on_SBM_2D"
        """ N = 500 """
        # "2020_02_16_15h56min03sec_infos_realizations" \
        # "_dictionary_for_winfree_on_SBM_2D"

    elif dictionary_name_string == "bipartite_kuramoto":
        realizations_dictionary_str = \
            "2020_02_19_20h39min44sec_infos_realizations_dictionary" \
            "_for_kuramoto_on_bipartite_2D_parameter_realizations_True"

        """ Target A, N = 250"""
        # "2020_02_16_23h22min31sec_infos_realizations" \
        # "_dictionary_for_kuramoto_on_bipartite_2D" \
        # "_parameter_realizations_False"
        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_15_18h00min03sec_infos_realizations" \
        # "_dictionary_for_kuramoto_on_bipartite_2D"
        """ N = 500 """
        # "2020_02_16_15h23min15sec_infos_realizations" \
        # "_dictionary_for_kuramoto_on_bipartite_2D"

    elif dictionary_name_string == "SBM_kuramoto":
        realizations_dictionary_str = \
            "2020_02_19_21h03min15sec_infos_realizations_dictionary" \
            "_for_kuramoto_on_SBM_2D_parameter_realizations_True"

        """ Target A, N = 250"""
        # "2020_02_16_23h28min49sec_infos_realizations" \
        # "_dictionary_for_kuramoto_on_SBM_2D" \
        # "_parameter_realizations_False"
        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_15_18h33min25sec_infos_" \
        # "realizations_dictionary_for_kuramoto_on_SBM_2D"
        """ N = 500 """
        # "2020_02_16_15h56min44sec_infos_realizations" \
        # "_dictionary_for_kuramoto_on_SBM_2D"

    elif dictionary_name_string == "bipartite_theta":
        realizations_dictionary_str = \
            "2020_02_19_20h39min46sec_infos_realizations_dictionary" \
            "_for_theta_on_bipartite_2D_parameter_realizations_True"

        """ Target A, N = 250"""
        # "2020_02_16_23h24min21sec_infos_realizations" \
        # "_dictionary_for_theta_on_bipartite_2D" \
        # "_parameter_realizations_False"
        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_16_22h14min08sec_infos_realizations" \
        # "_dictionary_for_theta_on_bipartite_2D"
        """ N = 500 """
        # "2020_02_16_15h22min34sec_infos_realizations" \
        # "_dictionary_for_theta_on_bipartite_2D"

    elif dictionary_name_string == "SBM_theta":
        realizations_dictionary_str = \
            "2020_02_19_21h02min36sec_infos_realizations" \
            "_dictionary_for_theta_on_SBM_2D_parameter_realizations_True"

        """ Target A, N = 250 """
        # "2020_02_16_23h29min26sec_infos_realizations" \
        # "_dictionary_for_theta_on_SBM_2D" \
        # "_parameter_realizations_False"
        # (below) With different parameters realizations
        """ N = 250 """
        # "2020_02_16_22h19min24sec_infos_realizations_" \
        # "dictionary_for_theta_on_SBM_2D"
        """ N = 500 """
        # "2020_02_16_15h55min49sec_infos_realizations" \
        # "_dictionary_for_theta_on_SBM_2D"

    else:
        raise ValueError("The given dictionary_name_string is not valid.")

    return realizations_dictionary_str  # (info_realizations_...)


def get_transitions_realizations_dictionary_absolute_path(
        dictionary_name_string):
    if dictionary_name_string == "bipartite_winfree":
        transitions_dictionary_str =  \
            "2020_02_19_20h39min45sec_multiple_synchro_transition" \
            "_dictionary_winfree_on_bipartite_2D_2020_02_20_10h48min43sec"
        # "2020_02_19_20h39min45sec_multiple_synchro_transition" \
        # "_dictionary_winfree_on_bipartite_2D_naive_" \
        # "2020_03_01_03h56min30sec"
        # "2020_02_16_23h19min21sec_multiple_synchro_transition" \
        # "_dictionary_winfree_on_bipartite_2D_2020_02_18_01h31min29sec"
        # "2020_02_15_18h01min36sec_multiple_synchro_transition" \
        # "_dictionary_winfree_on_bipartite_2D_2020_02_16_07h18min57sec"

    elif dictionary_name_string == "SBM_winfree":
        transitions_dictionary_str = \
            "2020_02_19_21h02min17sec_multiple_synchro_transition" \
            "_dictionary_winfree_on_SBM_2D_2020_02_21_05h55min00sec"
        # "2020_02_19_21h02min17sec_multiple_synchro_transition" \
        # "_dictionary_winfree_on_SBM_2D_naive_2020_03_01_03h56min30sec"
        # "2020_02_16_23h28min50sec_multiple_synchro_transition" \
        # "_dictionary_winfree_on_SBM_2D_2020_02_18_01h25min38sec"
        # "2020_02_15_18h33min11sec_multiple_synchro_transition" \
        # "_dictionary_winfree_on_SBM_2D_2020_02_16_14h42min11sec"

    elif dictionary_name_string == "bipartite_kuramoto":
        transitions_dictionary_str = \
            "2020_02_19_20h39min44sec_multiple_synchro_transition" \
            "_dictionary_kuramoto_on_bipartite_2D_2020_02_20_10h49min15sec"
        # "2020_02_16_23h22min31sec_multiple_synchro_transition" \
        # "_dictionary_kuramoto_on_bipartite_2D_2020_02_18_04h32min14sec"
        # "2020_02_15_18h00min03sec_multiple_synchro_transition" \
        # "_dictionary_kuramoto_on_bipartite_2D_2020_02_16_08h22min20sec"

    elif dictionary_name_string == "SBM_kuramoto":
        transitions_dictionary_str = \
            "2020_02_19_21h03min15sec_multiple_synchro_transition" \
            "_dictionary_kuramoto_on_SBM_2D_2020_02_21_05h55min41sec"
        # "2020_02_16_23h28min49sec_multiple_synchro_transition" \
        # "_dictionary_kuramoto_on_SBM_2D_2020_02_18_04h31min35sec"
        # "2020_02_15_18h33min25sec_multiple_synchro_transition" \
        # "_dictionary_kuramoto_on_SBM_2D_2020_02_16_15h08min52sec"

    elif dictionary_name_string == "bipartite_theta":
        transitions_dictionary_str = \
            "2020_02_19_20h39min46sec_multiple_synchro_transition" \
            "_dictionary_theta_on_bipartite_2D_2020_02_20_10h39min45sec"
        # "2020_02_16_23h24min21sec_multiple_synchro_transition" \
        # "_dictionary_theta_on_bipartite_2D_2020_02_18_01h21min11sec"
        # "2020_02_15_18h27min54sec_multiple_synchro_transition" \
        # "_dictionary_theta_on_bipartite_2D_2020_02_16_08h20min33sec"

    elif dictionary_name_string == "SBM_theta":
        transitions_dictionary_str = \
            "2020_02_19_21h02min36sec_multiple_synchro_transition" \
            "_dictionary_theta_on_SBM_2D_2020_02_21_05h54min22sec"
        # "2020_02_16_23h29min26sec_multiple_synchro_transition" \
        # "_dictionary_theta_on_SBM_2D_2020_02_18_01h31min50sec"
        # "2020_02_15_18h33min29sec_multiple_synchro_transition" \
        # "_dictionary_theta_on_SBM_2D_2020_02_16_14h47min39sec"

    else:
        raise ValueError("The given dictionary_name_string is not valid.")

    return transitions_dictionary_str


def global_reduction_matrix(M):
    """
    :param M: Reduction matrix
    :return: m: global reduction matrix
    """
    l_weights = np.count_nonzero(M, axis=1)
    l_normalized_weights = l_weights / np.sum(l_weights)
    return M.T @ l_normalized_weights


def get_matrix_of_flatten_reduction_matrices_for_all_targets(CVM_dictionary):

    """
    get_matrix_of_flatten_reduction_matrices_for_all_targets for principal
    component analysis
    :param CVM_dictionary: it is the resulting dictionary from the function
                           get_reduction_matrices_for_all_targets. See the
                           documentation of the function.
    :return:
    """
    M_K = np.array(CVM_dictionary["M_K"])
    n, N = np.shape(M_K)
    targets_possibilities = ["W", "WK", "WA", "WKA", "WAK",
                             "K", "KW", "KA", "KWA", "KAW",
                             "A", "AW", "AK", "AWK", "AKW"]
    flatten_reduction_matrices_array = np.zeros((n*N,
                                                 len(targets_possibilities)))
    for i, targets_string in enumerate(targets_possibilities):
        M = np.array(CVM_dictionary[f"M_{targets_string}"])
        flatten_reduction_matrices_array[:, i] = M.flatten()

    return flatten_reduction_matrices_array.T, targets_possibilities


def get_matrix_of_flatten_reduction_matrices_for_all_realizations(
        realizations_dictionary):

    """
    Get matrix of flatten reduction matrices for all realizations of a random
    graph and possibly random parameters.
    :param realizations_dictionary: it is the resulting dictionary from
                           the function get_realizations_dictionary. See the
                           documentation of the function.
    :return: A matrix that can be used for principal component analysis.
    """
    M_realizations = realizations_dictionary["M_realizations"]
    n, N = np.shape(M_realizations[0])
    flatten_reduction_matrices_array = np.zeros((n*N, len(M_realizations)))
    for i, M in enumerate(M_realizations):
        flatten_reduction_matrices_array[:, i] = np.array(M).flatten()

    return flatten_reduction_matrices_array.T


def get_reduced_parameter_matrix(M, X):
    return M@X@pinv(M)


# --------------------- Error functions and properties ------------------------

def mse(a, b):
    return np.mean((a - b)**2)


def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))


def nrmse(a, b):
    # TODO
    return a*b


def get_rmse_error(M, X):
    return rmse(M@X, M@X@pinv(M)@M)


def get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                    V_T1, V_T2, V_T3,
                                                    M, snmf_frobenius_error,
                                                    onmf_frobenius_error,
                                                    onmf_ortho_error,
                                                    W, K, A):
    """

    :param T1: First target
    :param T2: Second target
    :param T3: Third targer
    :param V_T1: Eigenvector matrix of the first target
    :param V_T2: Eigenvector matrix of the second target
    :param V_T3: Eigenvector matrix of the third target
    :param M: Reduction matrix
    :param snmf_frobenius_error: Frobenius error ||X - WH||
                                 for the factorization in the snmf
    :param onmf_frobenius_error: Frobenius error ||X - WH||
                                 for the factorization in the onmf
    :param onmf_ortho_error: Frobenius error ||I - HH^T||
                             for the orthogonality for H in the onmf
    # :param C_T1: Coefficent matrix for target 1
    #              without the normalization (it is related to C_T2 in a three
    #              target procedure. In a two target procedure, C_T2 is related
    #              to the normalization)
    # :param C_T2: Coefficent matrix without for target 1
    #              without the normalization (C_T3 would be the normalization
    #              matrix)
    :param W: Diagonal frequency matrix
    :param K: Diagonal degree matrix
    :param A: Adjacency matrix
    :return: properties: The properties and errors of the dimension-reduction
    """
    m = global_reduction_matrix(M)
    # Mp = pinv(M)
    redW = get_reduced_parameter_matrix(M, W)
    redK = get_reduced_parameter_matrix(M, K)
    redA = get_reduced_parameter_matrix(M, A)
    np.set_printoptions(precision=4)
    properties = f"\n---------------------- " \
                 f"T1 = {T1} -> T2 = {T2} -> T3 = {T3}"\
                 f"------------------------ \n" \
                 f"\n \n V_T1 = {V_T1}" \
                 f"\n \n V_T2 = {V_T2}" \
                 f"\n \n V_T3 = {V_T3}" \
                 f"\n \n M = {M}" \
                 f"\n \n m = {m}" \
                 f"\n \n redW = M W M^+ =\n{redW}"\
                 f"\n \n spec(redW) = {eig(redW)[0]}" \
                 f"\n \n redK = M K M^+ =\n{redK}" \
                 f"\n \n spec(redK) = {eig(redK)[0]}" \
                 f"\n \n redA = M A M^+ =\n{redA}" \
                 f"\n \n spec(redA) = {eig(redA)[0]}" \
                 f"\n \n snmf_frobenius_error = {snmf_frobenius_error}" \
                 f"\n \n onmf_frobenius_error = {onmf_frobenius_error}" \
                 f"\n \n onmf_ortho_error = {onmf_ortho_error}" \
                 f"\n \n np.sqrt(||M W - M W M^+M ||^2) = " \
                 f"{np.round(rmse(M@W, redW@M), 4)}\n" \
                 f"\n \n np.sqrt(||M K - M K M^+M ||^2) = " \
                 f"{np.round(rmse(M@K, redK@M), 4)}\n" \
                 f"\n \n np.sqrt(||M A - M A M^+M ||^2) = " \
                 f"{np.round(rmse(M@A, redA@M), 4)}\n" \
                 f"\n \n reduction_matrix_has_rank_n = " \
                 f"{matrix_has_rank_n(M)}" \
                 f"\n \n reduction_matrix_is_normalized = " \
                 f"{matrix_is_normalized(M)}" \
                 f"\n \n reduction_matrix_is_positive = " \
                 f"{matrix_is_positive(M)}" \
                 f"\n \n global reduction_matrix_is_normalized = " \
                 f"{matrix_is_normalized(m)}" \
                 f"\n \n global reduction_matrix_is_positive =" \
                 f" {matrix_is_positive(m)}" \
                 "\n --------" \
                 "-----------------------------------------------" \
                 "-----------------------\n\n\n\n"
    # "\n \n C_T1^+ C_T1 = {pinv(C_T1)@C_T1}\n" \
    # "\n \n C_T2^+ C_T2 = {pinv(C_T2)@C_T2} ([[0.]]
    # if not three target method)" \ return properties
    # f"\n \n M M_^+ = {M@Mp} \n" \
    # f"\n \n M^+ M = {Mp@M}\n" \

    return properties


# if __name__ == "__main__":
#     from sklearn.decomposition import *
#
#     M = np.array([[0.1132, 0.2582, 0.1471, 0.1771, 0.1524, 0.152],
#                   [0.1383, 0.1647, 0.0285, 0.2838, 0.1925, 0.1922],
#                   [0.022, 0.302, 0.2822, 0.0374, 0.1784, 0.178]])
#
#     model = NMF(n_components=3, init='nndsvd', random_state=0)
#     W = model.fit_transform(M)
#     H = model.components_
#
#     print(W, "\n\n", np.round(H, 3), "\n\n", W@H - M)
#
#     # print(compute_nmf(M, 2)
#     # X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]],
#     #              dtype=np.Py_ssize_t)
#     # import ristretto.nmf as ro
#     # W, H = ro.compute_nmf(X, rank=2)
#
#     import numpy as np
#     from ionmf.factorization.onmf import onmf
#     W, H = onmf(M, 3)
#
#     print(np.round(W, 3), "\n\n", np.round(H, 3), "\n\n", W @ H - M)

""" Old """
# TODO
# def get_non_negativity_factors_alpha(V):
#     """
#
#     :param V: V must be a a matrix n x N in which the elements of
#               the first line are all positive. The other lines can have
#               negative values.
#     :return: alpha: Nonnegativity factor, an array of length n that is used
#                     to get the nonnegativity matrix
#                     (see get_non_negativity_matrix_E)
#     """
#     #
#
#     if np.any(V[0, :]) < 0:
#       raise ValueError('V[0, :] as negative values. V must be a n x N matrix'
#                          ' in which\n the elements of the first line are '
#                          'all positive.')
#     else:
#         alpha = np.max(V/V[0, :], axis=1)
#
#     return alpha
#
#
# def get_non_negativity_matrix_E(V):
#     """
#
#     :param V: V must be a a matrix n x N in which the elements of
#     the first line are all positive. The other lines can have
#     negative values.
#     :return: E: n x n matrix use to obtain a non negative dimension-reduction
#                 matrix in a one target reduction with T1 = A
#     """
#     n = len(V[:, 0])
#     E = np.identity(n)
#     E[:, 0] = get_non_negativity_factors_alpha(V)
#     return E
# def get_reduction_matrix(C_T1, V):# TODO peut-être merger avec fct précédente
#     """
#     :param C_T1: Coefficent matrix for target 1
#                  without the normalization (it is related to C_T2
#                  [see get_second_target_coefficent_matrix] in a three
#                  target procedure. In a two target procedure, C_T2 is related
#                  to the normalization)
#     :param V_T1: First target eigenvector matrix
#
#     :return:
#     """
#     return ((C_T1@V).T / (np.sum(C_T1@V_T1, axis=1))).T
# if __name__ == '__main__':
#
#
#     A = two_triangles_graph_adjacency_matrix()
#     K = np.diag(np.sum(A, axis=1))
#     W = np.diag(np.array([0.2, 0.2, 0.2, -0.2, -0.2, -0.2]))
#
#     vapW = eig(W)[0]
#     vapK = eig(K)[0]
#     vapvep = eig(A)
#
#     V0 = -vapvep[1][:, 0]
#     # The minus sign is because python gives the dom. eigenvector
#     # with negative
#     # signs for each element.
#     V1 = vapvep[1][:, 1]
#     V2 = vapvep[1][:, 2]
#     V3 = vapvep[1][:, 3]
#     V4 = vapvep[1][:, 4]
#     V5 = vapvep[1][:, 5]
#
#     V_W_3D = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
#                        [0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
#                        [0, 0, 0, 0, 0, 1]])
#
#     V_K_3D = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0, 0],
#                        [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
#                        [0, 0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]])
#
#     V_A = np.array([V0, V1, V3])
#
#     C_K = get_second_target_coefficent_matrix(V_A, V_K_3D, V_W_3D,
#                                               other_procedure=False)
#
#     C_A = get_first_target_coefficent_matrix(C_K, V_A, V_K_3D)
#
#     M = get_reduction_matrix(C_A, V_A)
#
#     print(M)
# from synch_predictions.graphs.graph_spectrum import *
# import numpy as np
# from scipy.linalg import pinv
# from numpy.linalg import multi_dot
# import networkx as nx
# import matplotlib.pyplot as plt
#
#
# def get_M_bipartite(sizes, p_out):
#
#     """
#     TO BE UPDATED !!!
#     :param sizes:
#     :param p_out:
#     :return:
#     """
#
#     pq = [[0, p_out], [p_out, 0]]
#     A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
#     V = get_eigenvectors_matrix(A, 2)  # Not normalized
#     K = np.diag(np.sum(A, axis=1))
#
#     m = np.array([V[0] - V[1], V[0] + V[1]])
#     M = (m.T / (np.sum(m, axis=1))).T
#
#     n1, n2 = sizes
#     Vp = pinv(V)
#     C = np.dot(np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
#                          [np.zeros(n1), 1 / n2 * np.ones(n2)]]), Vp)
#     print(C, "\n")
#     CV = np.dot(C, V)
#     M_block = (CV.T / (np.sum(CV, axis=1))).T
#     P = (np.block([[np.ones(n1), np.zeros(n2)],
#                    [np.zeros(n1), np.ones(n2)]])).T
#     print(multi_dot([M, K, pinv(M)]), "\n")
#     print(multi_dot([M_block, K, pinv(M_block)]), "\n")
#     print((multi_dot([M_block**2, A, P]).T /
#            np.diag(np.dot(M_block, M_block.T))).T, "\n")
#
#     return M, M_block
#
#
# def get_block_M_bipartite(M_T, sizes, p_out):
#     pq = [[0, p_out], [p_out, 0]]
#     A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
#     V = get_eigenvectors_matrix(A, 2)  # Not normalized
#     Vp = pinv(V)
#     C = np.dot(M_T, Vp)
#     print(C)
#     CV = np.dot(C, V)
#     M = (CV.T / (np.sum(CV, axis=1))).T
#     return M
#
#
# M, M_block = get_M_bipartite([5, 5], 0.5)
# print(M, M_block, np.mean(M[0, 0:50]), np.std(M[0, 0:50]))
#
# plt.figure(figsize=(5, 5))
# ax1 = plt.subplot(211)
# ax1.matshow(M, aspect="auto")
# ax2 = plt.subplot(212)
# ax2.matshow(M_block, aspect="auto")
# plt.show()
#
# # n1, n2 = 50, 50
# # plt.matshow(get_block_M_bipartite(np.block([[1 / n1 * np.ones(n1),
# #  np.zeros(n2)], [np.zeros(n1), 1 / n2 * np.ones(n2)]]) ,[50, 50], 0.5),
# #  aspect="auto")
# # plt.show()
