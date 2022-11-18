# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


from dynamics.integrate import *
from dynamics.dynamics import *
from dynamics.reduced_dynamics import *
from graphs.graph_spectrum import *
from plots.plot_complete_vs_reduced_old import *
from plots.plots_setup import *
# from synchro_integration import give_pq
import numpy as np
from numpy.linalg import multi_dot, pinv


# Time parameters
t0, t1, dt = 0, 20, 0.0001
timelist = np.arange(t0, t1, dt)


# Graph parameters
n1 = 100
n2 = 100
sizes = [n1, n2]
N = sum(sizes)
beta = (n1*(n1-1) + n2*(n2-1)) / (N*(N-1))
# rho = 0.5
# delta = 0.2
# pq = give_pq(rho, delta, beta)
pin = 0.8
pout = 0.1
pq = [[pin, pout], [pout, pin]]
# A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
A = np.zeros((N, N))
ii = 0
for i in range(0, len(sizes)):
    jj = 0
    for j in range(0, len(sizes)):
        A[ii:ii + sizes[i], jj:jj + sizes[j]] \
            = pq[i][j]*np.ones((sizes[i], sizes[j]))
        jj += sizes[j]
    ii += sizes[i]
k_array = np.dot(A, np.ones(len(A[:, 0])).transpose())
K = np.diag(k_array)
# V = get_eigenvectors_matrix(A, 2)  # get_eigenvectors_matrix(A, 1)[0]
# P = np.block([[np.ones((n1, 1)), np.zeros((n1, 1))],
#               [np.zeros((n2, 1)), np.ones((n2, 1))]])
# M = np.array([V[0, :]/np.sum(V[0, :]),
#               np.absolute(V[1, :])/np.sum(np.absolute(V[1, :]))])
# #M = np.array([V[0, :]/np.sum(V[0, :]),
# #          np.absolute(V[1, :])/np.sum(np.absolute(V[1, :]))])
# #M = np.array([np.concatenate([V[0:n1]/np.sum(V[0:n1]), np.zeros(n2)]),
# #          np.concatenate([np.zeros(n1), V[n1:]/np.sum(V[n1:])])])

V = get_eigenvectors_matrix(A, 2)  # Not normalized
M_0 = np.block([[1/n1*np.ones(n1), np.zeros(n2)],
                [np.zeros(n1), 1/n2*np.ones(n2)]])
# M_0 = np.block([[V[0,0:n1], np.zeros(n2)],
#                 [np.zeros(n1), V[0, n1:]]])
P = (np.block([[np.ones(n1), np.zeros(n2)],
               [np.zeros(n1), np.ones(n2)]])).T
Vp = pinv(V)
C = np.dot(M_0, Vp)
CV = np.dot(C, V)
# M = (CV.T/(np.sum(CV, axis=1))).T
# print(np.sum(M, axis=1))
M = M_0

redA = multi_dot([M, A, P])
# - np.array([[60, multi_dot([M, A, P])[0, 1]-20]
#            ,[multi_dot([M, A, P])[1, 0]-25, 40]])
hatredA = multi_dot([M, A, M.transpose()])/np.diag(np.dot(M**2, P))
redK = multi_dot([M, A, P])
hatredK = (multi_dot([M**2, A, P]).transpose()
           / np.sum(M**2, axis=1)).transpose()

print("redA =", redA, "\n", "\n", "hatredA =", hatredA, "\n", "\n",
      "redK =", redK, "\n", "\n", "hatredK =", hatredK, "\n", "\n", )

# Dynamical parameters
sigma = 8
Iext = -1
theta0 = 2*np.pi*np.random.rand(N)  # np.linspace(0, 2 * np.pi, N)  #
z0 = np.exp(1j * theta0)
Z0 = np.dot(M, z0)


# Integration
args_theta = (Iext, sigma)
theta_sol = integrate_dynamics(t0, t1, dt, theta, A, "vode", theta0,
                               *args_theta)
r1t_theta = np.absolute(np.sum(M[0, :] * np.exp(1j * theta_sol), axis=1))
r2t_theta = np.absolute(np.sum(M[1, :] * np.exp(1j * theta_sol), axis=1))

args_Z_theta = (Iext, sigma, N, redK, redA, redK)
Z_theta_sol = integrate_dynamics(t0, t1, dt, reduced_theta, redA, "zvode", Z0,
                                 *args_Z_theta)
R1t_theta = np.absolute(Z_theta_sol[:, 0])
R2t_theta = np.absolute(Z_theta_sol[:, 1])


# Plot
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True

spike_array = (1-np.cos(theta_sol)).transpose()
for j in range(0, len(timelist)):
    for i in range(0, N):
        if spike_array[i, j] < 1.5:
            spike_array[i, j] = 0
plt.matshow(spike_array, aspect="auto")
plt.colorbar()
plt.show()


plot_complete_vs_reduced_vs_time(timelist, r1t_theta, np.ones(len(timelist)),
                                 R1t_theta, np.ones(len(timelist)))
plt.show()
