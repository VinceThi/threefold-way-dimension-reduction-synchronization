import networkx as nx
import numpy as np


def two_star_graph_adjacency_matrix(sizes, pq):
    """

    :param sizes: Sizes of the stars (core + nb of leaves = 1 + nf)
    :param pq = [[s1, scc],[scc, s2]] where s1, s2 and scc are weights between
           the first periphery and the first core, the second periphery and
           the second core and the two cores respectively
    :return:  The weighted adjacency matrix of the two-star graph
    """
    N = sum(sizes)
    A = np.zeros((N, N))
    ii = 0
    for i in range(0, len(sizes)):
        jj = 0
        for j in range(0, len(sizes)):
            if i == j:
                A[ii:ii + sizes[i], jj:jj + sizes[j]] \
                    = pq[i][j] * nx.to_numpy_matrix(
                    nx.star_graph(sizes[i]-1, create_using=None))
            else:
                A[ii, jj] = pq[i][j]
            jj += sizes[j]
        ii += sizes[i]
    return A


def two_triangles_graph_adjacency_matrix():
    A = np.array([[0, 1, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [1, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 1],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1, 0]])
    return A


def small_bipartite_graph_adjacency_matrix():
    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0]])
    return A


def mean_SBM(sizes, pq, self_loop=False):
    """
    :param sizes:
    :param pq:
    :param self_loop:
    :return:
    """
    p_mat_up = pq[0][0] * np.ones((sizes[0], sizes[0]))
    p_mat_low = pq[1][1] * np.ones((sizes[1], sizes[1]))
    q_mat_up = pq[0][1] * np.ones((sizes[0], sizes[1]))
    q_mat_low = pq[1][0] * np.ones((sizes[1], sizes[0]))

    mA = np.block([[p_mat_up, q_mat_up], [q_mat_low, p_mat_low]])

    if self_loop:
        mean_A = mA
    else:
        if not np.all(np.diag(mA)):
            mean_A = mA - np.diag(mA)
        else:
            mean_A = mA

    return mean_A
