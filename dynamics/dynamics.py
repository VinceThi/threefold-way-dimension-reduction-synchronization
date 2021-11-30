# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


import numpy as np
from numba import jit


# @jit(nopython=True)
def kuramoto(t, theta, A, omega, sigma):
    """

    :param t:
    :param theta:
    :param omega:
    :param sigma:
    :param A:
    :param N:
    :return:
    """
    N = len(A[:, 0])
    return omega + sigma/N*(np.cos(theta)*(A@np.sin(theta))
                            - np.sin(theta)*(A@np.cos(theta)))


# @jit(nopython=True)
def kuramoto_sakaguchi(t, theta, A, omega, sigma, alpha):
    """

    :param t:
    :param theta:
    :param omega:
    :param sigma:
    :param A:
    :param alpha:
    :param N:
    :return:
    """
    N = len(A[:, 0])
    return omega \
        + sigma/N*(np.cos(theta+alpha)*np.dot(A, np.sin(theta))
                   - np.sin(theta+alpha)*np.dot(A, np.cos(theta)))


# @jit(nopython=True)
def theta(t, theta, A, Iext, sigma):
    """

    :param t:
    :param theta:
    :param Iext:
    :param sigma:
    :param A:
    :param N:
    :param kappa:
    :return:
    """
    N = len(A[:, 0])
    return 1 - np.cos(theta) \
        + (1 + np.cos(theta))*(Iext + sigma/N*(A@(np.ones(N)-np.cos(theta))))


# @jit(nopython=True)
def winfree(t, theta, A, omega, sigma):
    """

    :param t:
    :param theta:
    :param omega:
    :param sigma:
    :param A:
    :param N:
    :return:
    """
    N = len(A[:, 0])
    return omega - sigma/N*np.sin(theta)*(A@(np.ones(N) + np.cos(theta)))


@jit(nopython=True)
def lorenz(t, X, A, omega, sigma, a, b, c):
    # omega not used finally
    N = len(A[0])
    x, y, z = X[0:N], X[N: 2*N], X[2*N:3*N]
    # print(len(x), len(y), len(z))
    dxdt = a*(y - x) - sigma*(x*np.sum(A, axis=1) - A@x)
    dydt = (b*x - y - x*z)
    dzdt = (x*y - c*z)
    return np.concatenate((dxdt, dydt, dzdt))


def rossler(t, X, A, omega, sigma, a, b, c):
    # omega not used finally
    N = len(A[0])
    x, y, z = X[0:N], X[N: 2*N], X[2*N:3*N]
    dxdt = (-y - z) - sigma*(x*np.sum(A, axis=1) - A@x)
    dydt = (x + a*y)
    dzdt = (b - c*z + z*x)
    return np.concatenate((dxdt, dydt, dzdt))


# @jit(nopython=True)
def cowan_wilson(t, x, A, tau, mu):
    return -x + A@(1/(1+np.exp(-tau*(x-mu))))


def wilson_cowan(t, x, W, a, b):
    return -x + 1/(1+np.exp(-a*(W@x-b)))


# def wilson_cowan_BCM(t, X, D, taux, tauw, taut, alpha, beta, gamma,
#                      a, b, c, xmax):
#     Dflat = D.flatten()
#     N = len(D[0])
#     x, w, theta = X[0:N], X[N: N+N**2], X[N+N**2:2*N+N**2]
#     y = np.reshape(w, (N, N))@x + gamma
#     dxdt = (-alpha*x + beta/(1+np.exp(-a*(y - b))))/taux
#     dwdt = (Dflat*np.tile(x, N)*np.repeat(x*(x-theta), N) - c*w)/tauw
#     dthetadt = (x**2 - theta*xmax)/taut
#     return np.concatenate((dxdt, dwdt, dthetadt))

def wilson_cowan_hebb(t, X, D, gamma, a, b, c):
    Dflat = D.flatten()
    N = len(D[0])
    x, w = X[0:N], X[N: N+N**2]
    y = np.reshape(w, (N, N))@x + gamma
    dxdt = -x + 1/(1+np.exp(-a*(y - b)))
    dwdt = Dflat*np.tile(x, N)*np.repeat(x, N) - c*w
    return np.concatenate((dxdt, dwdt))


def wilson_cowan_oja(t, X, D, gamma, a, b, c):
    Dflat = D.flatten()
    N = len(D[0])
    x, w = X[0:N], X[N: N+N**2]
    y = np.reshape(w, (N, N))@x + gamma
    dxdt = -x + 1/(1+np.exp(-a*(y - b)))
    dwdt = Dflat*np.tile(x, N)*np.repeat(x, N) - c*w*np.repeat(x**2, N)
    return np.concatenate((dxdt, dwdt))


def wilson_cowan_BCM(t, X, D, tautx, gamma, a, b, c, xmax):
    Dflat = D.flatten()
    N = len(D[0])
    x, w, theta = X[0:N], X[N: N+N**2], X[N+N**2:2*N+N**2]
    y = np.reshape(w, (N, N))@x + gamma
    dxdt = -x + 1/(1+np.exp(-a*(y - b)))
    dwdt = Dflat*np.tile(x, N)*np.repeat(x*(x-theta), N) - c*w
    dthetadt = (x**2 - theta*xmax)/tautx
    return np.concatenate((dxdt, dwdt, dthetadt))


def cosinus(t, theta, A, omega, sigma):
    """

    :param t:
    :param theta:
    :param omega:
    :param sigma:
    :param A:
    :param N:
    :return:
    """
    N = len(A[:, 0])
    return omega + sigma/N*np.dot(A, np.cos(theta))


def hopf(t, z, K, omega, N):
    return (1 + 1j*omega - np.absolute(z)**2)*z + K/N*np.sum(z) - K*z
