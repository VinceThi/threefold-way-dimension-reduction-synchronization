# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


from synchro_integration import give_pq
import numpy as np
from synch_predictions.graphs.graph_spectrum import *
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import pinv


# Graph parameters
n1 = 50
n2 = 50
sizes = [n1, n2]
N = sum(sizes)
beta = (n1*(n1-1) + n2*(n2-1)) / (N*(N-1))
# rho = 0.4
# delta = 0.2
# pq = give_pq(rho, delta, beta)
pin = 0
pout = 0.1
pq = [[pin, pout], [pout, pin]]

############################### Adjacency matrix  #############################
A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
#"""
V = get_eigenvectors_matrix(A, 2)

# plt.plot(V[0])
# plt.show()

M_0 = np.block([[1/n1*np.ones(n1), np.zeros(n2)],
                [np.zeros(n1), 1/n2*np.ones(n2)]])
Vp = pinv(V)
C = np.dot(M_0, Vp)

plt.matshow(C, aspect="auto")
plt.colorbar()
plt.show()

CV = np.dot(C, V)

# Not normalized
# M = (CV.T / (np.sum(np.abs(CV), axis=1))).T # Negative weigths ! Watch out !

# Normalization 1
M_norm1 = (np.abs(CV).T / (np.sum(np.abs(CV), axis=1))).T

# Normalization 2
M_min_max_norm = ((CV.T - np.min(CV, axis=1))/(
        np.max(CV, axis=1) - np.min(CV, axis=1))).T
M_norm_to_sum_1 = (M_min_max_norm.T / (np.sum(M_min_max_norm, axis=1))).T

# Normalization 3
M_norm3 = (CV.T / (np.sum(CV, axis=1))).T


matrix_list = [M_norm1, M_norm_to_sum_1, M_norm3]
fig, axes = plt.subplots(nrows=3, ncols=1)
i = 0
for ax in axes.flat:
    im = ax.imshow(matrix_list[i], cmap=plt.get_cmap('Greys'), aspect="auto")
    i += 1
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()
#"""

############################### Laplacian matrix  #############################
"""
k_array = np.dot(A, np.ones(len(A[:, 0])).transpose())
K = np.diag(k_array)
L = K - A
V = get_laplacian_eigenvectors_matrix(L)
M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                [np.zeros(n1), 1 / n2 * np.ones(n2)]])
Vp = pinv(V)
C = np.dot(M_0, Vp)
CV = np.dot(C, V)

print(np.sum(V, axis=1))
# Not normalized
# M = (CV.T / (np.sum(np.abs(CV), axis=1))).T # Negative weigths ! Watch out !

print(V)

plt.plot(V[0])
plt.show()

matrix_list = [V, V]
fig, axes = plt.subplots(nrows=2, ncols=1)
i = 0
for ax in axes.flat:
    im = ax.imshow(matrix_list[i], cmap=plt.get_cmap('Greys'), aspect="auto")
    i += 1
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()
"""

# Other simple way, less flexible
# plt.matshow(V, aspect="auto")
# plt.colorbar()
# plt.show()
