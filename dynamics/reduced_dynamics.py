# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numba import jit


# @jit(nopython=True)
def reduced_theta_complex(t, Z, MAMp, MWMp, MKMp, sigma, N, kappa, Omega):
    return -1j/2*(Z - 1)**2 + 1j*Omega/2*((MWMp@Z)/Omega + 1)**2 \
           + 0.25*1j*sigma/N*((MKMp@Z)/kappa + 1)**2 * \
           (2*kappa - MAMp@(Z + np.conj(Z)))


# @jit(nopython=True)
def reduced_winfree_complex(t, Z, MAMp, MWMp, MKMp, sigma, N, kappa, Omega):
    return 1j*MWMp@Z + 0.5*kappa*sigma/N \
           + 0.25*sigma/N*(MAMp@(Z + np.conj(Z))) \
           - 0.5*sigma/N*kappa**(-1)*(MKMp@Z)**2 \
           - 0.25*sigma/N*kappa**(-2)*(MAMp@(Z + np.conj(Z)))*(MKMp@Z)**2


# @jit(nopython=True)
def reduced_kuramoto_complex(t, Z, MAMp, MWMp, MKMp, sigma, N, kappa, Omega):
    """
    Reduced Kuramoto dynamics
    :param t:
    :param Z:
    :param MAMp:
    :param MWMp:
    :param MKMp:
    :param sigma:
    :param N:
    :param kappa:
    :return:
    """
    return 1j*MWMp@Z + 0.5*sigma/N*MAMp@Z \
        - 0.5*sigma/N*kappa**(-2)*(MAMp@np.conj(Z))*(MKMp@Z)**2


def reduced_kuramoto_sakaguchi_complex(t, Z, MAMp, MWMp, MKMp,
                                       sigma, N, kappa, Omega, alpha):
    """
    Reduced Kuramoto dynamics
    :param t:
    :param Z:
    :param MAMp:
    :param MWMp:
    :param MKMp:
    :param sigma:
    :param N:
    :param kappa:
    :return:
    """
    return 1j*MWMp@Z + 0.5*sigma/N*np.exp(-1j*alpha)*MAMp@Z \
        - 0.5*sigma/N*kappa**(-2)*np.exp(-1j*alpha)*\
           (MAMp@np.conj(Z))*(MKMp@Z)**2


def reduced_kuramoto_sakaguchi(t, Z, reduced_A, reduced_K, reduced_W,
                               coupling, alpha, N):
    """
    Function to integrate the Sakaguchi-Kuramoto dynamics with ode.
    The coupling term is given by coupling/N. In the formalism of the
    threefold dimension reduction

    :param w:
    :param t:
    :param reduced_A: With no D_alpha
    :param reduced_K: With D_alpha
    :param reduced_W:
    :param alpha:
    :return:

    The shape of these matrix is (q, numberoftimepoints).
    q is the reduced dimension (number of blocks)
    """
    return 1j*np.dot(reduced_W, Z) \
        + 0.5*coupling/N*np.exp(-1j*alpha)*np.dot(reduced_A, Z) \
        - 0.5*coupling/N*np.exp(1j*alpha)*np.conj(np.dot(reduced_A, Z))\
        * (np.dot(reduced_K, Z))**2

# ---------------------------- Theta ------------------------------------------


def reduced_theta_1D(t, Z, kappa, Iext, sigma, hatkappa, N):
    # Old reduction
    return -1j/2*(Z-1)**2 + Iext*1j/2*(Z + 1)**2 \
        + 1j*sigma/(4*N)*(Z + 1)**2*kappa*(2 - Z - np.conj(Z)) \
        + 1j*sigma/(2*N)*(Z + 1)*(hatkappa-kappa)*Z*(2 - Z - np.conj(Z))


def reduced_theta_2D(t, Z, redA, I_array, sigma, N, hatredA, hatLambda):
    # Old reduction
    return -1j/2*(Z-1)**2 + 1j/2*(Z+1)**2*I_array \
           + 1j*sigma/(2*N)*(Z+1)**2*np.dot(redA, np.ones(len(Z))
                                            - 0.5*Z - 0.5*np.conj(Z)) \
           + 1j*sigma/N*Z*(Z+1)*np.dot(hatredA-redA, np.ones(len(Z))
                                       - 0.5*Z - 0.5*np.conj(Z)) \
           - 1j*sigma/(2*N)*(Z+1)**2*np.dot(hatLambda-redA,
                                            0.5*Z + 0.5*np.conj(Z))


def reduced_theta_R_Phi_2D(t, W, redA, I_array, sigma, N):
    R, Phi = W[0:2], W[2:4]

    dRdt = 0.5*(1-R**2)*np.sin(Phi)*(I_array - np.ones(2) +
                                     sigma/N*np.dot(redA, np.ones(2)
                                                    - R*np.cos(Phi)))
    dPhidt = np.ones(2) - (1+R**2)/(2*R)*np.cos(Phi) + \
        (np.ones(2) + (1+R**2)/(2*R)*np.cos(Phi)) * \
        (I_array + sigma/N*np.dot(redA, np.ones(2) - R*np.cos(Phi)))

    return np.concatenate([dRdt, dPhidt])


# def reduced_theta(t, Z, redA, Iext, sigma, N, redK, hatredA, hatredK):
#     return -1j/2*(Z-1)**2 + 1j/2*(Z+1)**2*Iext \
#            + 1j*sigma/(2*N)*(Z+1)**2*np.dot(redA, np.ones(len(Z))
#                                             + 0.5*Z + 0.5*np.conj(Z)) \
#            + 1j*sigma/N*Z*(Z+1)*np.dot(np.abs(hatredK)
#                                        - np.abs(redK), np.ones(len(Z))
#                                        + 0.5*Z + 0.5*np.conj(Z)) \
#            + 1j*sigma/(4*N)*(Z+1)**2*np.dot(np.abs(hatredA)-np.abs(redA),
#                                             Z*(1 - np.conj(Z)**2))
#
""""""

# ---------------------------- Winfree ----------------------------------------


def reduced_winfree_2D2(t, Z, MAMp, MWMp, sigma, N, MKMp, alpha):
    return 1j*np.dot(MWMp, Z) + 0.5*alpha/N \
           + 0.25/N*np.dot(MAMp, Z + np.conj(Z)) \
           - 0.5/N*alpha**(-1)*(np.dot(MKMp, Z))**2 \
           - 0.25/N*alpha**(-2)*(np.dot(MAMp, Z + np.conj(Z)))\
           * (np.dot(MKMp, Z))**2


def reduced_winfree_2D(t, Z, redA, omega_array, sigma, N, hatredA, hatLambda):
    # Old reduction
    return 1j*omega_array*Z - 0.5*sigma/N*(Z**2-1)*np.dot(
           redA, (np.ones(len(Z))+0.5*Z+0.5*np.conj(Z))) \
           - sigma/N*Z**2*np.dot(
           (hatredA - redA), (np.ones(len(Z))+0.5*Z+0.5*np.conj(Z))) \
           - 0.5*sigma/N*(Z**2-1)*np.dot(
           (hatLambda - redA), (0.5*Z+0.5*np.conj(Z)))


def reduced_winfree_R_Phi_2D(t, W, redA, omega_array, sigma, N):
    R, Phi = W[0:2], W[2:4]

    dRdt = 0.5*sigma*(1-R**2)/N*np.cos(Phi)*np.dot(redA,
                                                   np.ones(2) + R*np.cos(Phi))
    dPhidt = omega_array - sigma*(1+R**2)/(2*N*R)*np.sin(Phi) * \
        np.dot(redA, np.ones(2) + R*np.cos(Phi))
    return np.concatenate([dRdt, dPhidt])


# ------------------------- Kuramoto ------------------------------------------

def reduced_kuramoto_2D2(t, Z, MAMp, MWMp, sigma, N, MKMp, alpha):
    """
    This function was used a lot, but it's better to integrate the real
    equations !
    :param t:
    :param Z:
    :param MAMp:
    :param MWMp:
    :param sigma:
    :param N:
    :param MKMp:
    :param alpha:
    :return:
    """
    return 1j*np.dot(MWMp, Z) + 0.5*sigma/N*np.dot(MAMp, Z) \
           - 0.5*sigma/N*alpha**(-2)*np.dot(MAMp, np.conj(Z)) \
           * np.dot(MKMp, Z)*np.dot(MKMp, Z)


# def reduced_kuramoto_R_Phi(t, W, MAMp, MWMp, MKMp, sigma, N, kappa):
#     """
#     TODO dynamique à corriger !
#     Reduced Kuramoto dynamics (2n dimension) in R_mu and Phi_mu  where
#     mu in {1,..., n}.
#
#     :param t: Time
#     :param W: (array of shape (n,)) Initial conditions
#     :param MAMp: (array of shape (n, n)) reduced adjacency matrix
#     :param MWMp: (array of shape (n, n)) reduced frequency matrix
#     :param MKMp: (array of shape (n, n)) reduced degree matrix
#     :param sigma: (float) coupling constant
#     :param N: (int) Number of nodes
#     :param kappa: (array of shape (n,)) Weighted degree equal to M@A
#
#     :return: np.concatenate([dRdt, dPhidt]): one step of integration
#     """
#     n = len(MAMp[0])
#     R, Phi = W[0:n], W[n:2*n]
#
#     R = np.reshape(R, (n, 1))
#     Phi = np.reshape(Phi, (n, 1))
#     kappa = np.reshape(kappa, (n, 1))
#
#     Phim = Phi - Phi.T
#     Phip = Phi + Phi.T
#
#     dRdt = (MWMp*np.sin(Phim) + 0.5*sigma/N*MAMp*np.cos(Phim))@R \
#         - 0.5*sigma/N*kappa**(-2)*(
#                    ((MAMp*np.cos(Phip))@R)*((MKMp@(R*np.cos(Phi)))**2
#                                             - (MKMp@(R*np.sin(Phi)))**2)
#                    + 2*((MAMp*np.sin(Phip))@R)
#                    * (MKMp@(R*np.sin(Phi)))*(MKMp@(R*np.cos(Phi))))
#
#     dPhidt = ((MWMp*np.cos(Phim) - 0.5*sigma/N*MAMp*np.sin(Phim))@R
#               - 0.5*sigma/N*kappa**(-2)*(
#                    ((MAMp*np.sin(Phip))@R)*((MKMp@(R*np.cos(Phi)))**2
#                                             - (MKMp@(R*np.sin(Phi)))**2)
#                    - 2*((MAMp*np.cos(Phip))@R)
#                    * (MKMp@(R*np.sin(Phi)))*(MKMp@(R*np.cos(Phi)))))/R
#
#     dRdt, dPhidt = np.reshape(dRdt, (n,)), np.reshape(dPhidt, (n,))
#
#     return np.concatenate([dRdt, dPhidt])


# This next function is an old reduction
def reduced_kuramoto_2D(t, Z, redA, omega_array, sigma, N, hatredA, hatLambda):
    # Old reduction
    return 1j*omega_array*Z + 0.5*sigma/N*np.dot(hatLambda, Z) \
           - 0.5*sigma/N*Z**2*np.dot(hatLambda, np.conj(Z)) \
           - sigma/N*Z**2*np.dot(hatredA - redA, np.conj(Z))


def reduced_kuramoto_star_2D(t, W, MAMp, N, omega1, omega2, coupling):
    """

        :param W: vector containing the variable of the problem
                  (Rp, Phi)
        :param t: time list
        :param N: Number of nodes
        :param omega1: natural frequency of the core
        :param omega2: natural frequency of the periphery
        :param coupling: sigma/N
        :return:
    """
    Rp = W[0]
    Phi = W[1]

    Nf = N - 1

    # Synchro of the periphery
    dRpdt = coupling*(1 - Rp**2)/2*np.cos(Phi)

    # Phase difference between the phase of the core and the mesoscopic phase
    # observable of the periphery
    dPhidt = omega1 - omega2 - coupling*(1+(N+Nf)*Rp**2)/(2*Rp)*np.sin(Phi)

    return np.array([dRpdt, dPhidt])


def reduced_kuramoto_star_3D(t, W, MAMp, N, omega1, omega2, coupling):
    """

        :param W: vector containing the variable of the problem
                  (Rp, Phi)
        :param t: time list
        :param N: Number of nodes
        :param omega1: natural frequency of the core
        :param omega2: natural frequency of the periphery
        :param coupling: sigma/N
        :return:
    """
    Rp = W[0]
    Phi1 = W[1]
    Phi2 = W[2]

    Nf = N - 1

    dRpdt = coupling*(1 - Rp**2)/2*np.cos(Phi1 - Phi2)
    dPhi1dt = omega1 + coupling*Nf*np.sin(Phi2 - Phi1)
    dPhi2dt = omega2 + coupling*((1+Rp**2)/(2*Rp))*np.sin(Phi1 - Phi2)

    return np.array([dRpdt, dPhi1dt, dPhi2dt])


def reduced_kuramoto_sakaguchi_star_2D(t, W, MAMp, N,
                                       omega1, omega2, coupling, alpha):
    """

        :param W: vector containing the variable of the problem
                  (Rp, Phi)
        :param t: time list
        :param N: Number of nodes
        :param omega1: natural frequency of the core
        :param omega2: natural frequency of the periphery
        :param coupling: sigma/N
        :return:
    """
    Rp = W[0]
    Phi = W[1]

    Nf = N - 1

    # Synchro of the periphery
    dRpdt = coupling*(1 - Rp**2)/2*np.cos(Phi - alpha)

    # Phase difference between the phase of the core and the mesoscopic phase
    # observable of the periphery
    dPhidt = omega1 - omega2 - coupling*(N - 1)*Rp*np.sin(Phi + alpha) \
        - coupling*(1+Rp**2)/(2*Rp)*np.sin(Phi - alpha)

    return np.array([dRpdt, dPhidt])


def reduced_kuramoto_sakaguchi_star_3D(t, W, MAMp, N,
                                       omega1, omega2, coupling, alpha):
    """

        :param W: vector containing the variable of the problem
                  (Rp, Phi)
        :param t: time list
        :param N: Number of nodes
        :param omega1: natural frequency of the core
        :param omega2: natural frequency of the periphery
        :param coupling: sigma/N
        :return:
    """
    Rp = W[0]
    Phi1 = W[1]
    Phi2 = W[2]

    Nf = N - 1

    dRpdt = coupling*(1 - Rp**2)/2*np.cos(Phi1 - Phi2 - alpha)
    dPhi1dt = omega1 + coupling*Nf*np.sin(Phi2 - Phi1 - alpha)
    dPhi2dt = omega2 + coupling*((1+Rp**2)/(2*Rp))*np.sin(Phi1 - Phi2 - alpha)

    return np.array([dRpdt, dPhi1dt, dPhi2dt])


def reduced_kuramoto_twostars(t, W, s1, s2, scc, nf1, nf2, coupling, alpha):
    """

    :param W: vector containing the variable of the problem
              (Rp1, Rp2, Phi1, Phi2, Phi12)
    :param t: time list
    :param omega: natural frequency
    :param s1, s2: Weight of the links between the core and the periphery
                   of the first and second stars respectively
    :param scc: Weight of the link between the two cores
    :param nf1, nf2: Size of the periphery of the first and second stars
                     respectively
    :param coupling: sigma/N
    :param alpha: Phase lag
    :return:"""
    # Rc1 = w[0]
    Rp1 = W[0]
    # Rc2= w[2]
    Rp2 = W[1]
    Phi1 = W[2]
    Phi2 = W[3]
    Phi12 = W[4]

    # dRc1dt = coupling * ((1 - Rc1 ** 2) / 2)*(nf1*s1*Rp1*np.cos(Phi1-alpha)
    #  + scc*Rc2*np.cos(Phi12-alpha))
    dRp1dt = coupling*(1 - Rp1 ** 2) / 2 * s1 * 1 * np.cos(Phi1 + alpha)
    # dRc2dt = coupling * ((1 - Rc2 ** 2) / 2)*(nf2*s2*Rp2*np.cos(Phi2-alpha)
    #   + scc*Rc1*np.cos(Phi12+alpha))
    dRp2dt = coupling*(1 - Rp2 ** 2) / 2 * s2 * 1 * np.cos(Phi2 + alpha)

    dPhi1dt = -coupling*(((1 + Rp1**2)/(2*Rp1))*s1*1*np.sin(Phi1 + alpha)
                         + nf1 * s1 * Rp1 * np.sin(Phi1 - alpha)
                         + scc * np.sin(Phi12 - alpha))
    dPhi2dt = -coupling*(((1 + Rp2**2)/(2*Rp2))*s2*1*np.sin(Phi2 + alpha)
                         + nf2 * s2 * Rp2 * np.sin(Phi2 - alpha)
                         - scc * np.sin(Phi12 + alpha))
    dPhi12dt = coupling*(nf2 * s2 * Rp2 * np.sin(Phi2 - alpha)
                         - scc * 1 * np.sin(Phi12 + alpha)
                         - nf1 * s1 * Rp1 * np.sin(Phi1 - alpha)
                         - scc * 1 * np.sin(Phi12 - alpha))
    # dRc1dt, dRc2dt
    return np.concatenate([dRp1dt, dRp2dt, dPhi1dt, dPhi2dt, dPhi12dt])


def reduced_kuramoto_R_Phi_2D(t, W, redA, omega_array, sigma, N):
    R, Phi = W[0:2], W[2:4]

    dRdt = 0.5*sigma*(1-R**2)/N*(np.cos(Phi)*np.dot(redA, R*np.cos(Phi))
                                 + np.sin(Phi)*np.dot(redA, R*np.sin(Phi)))
    dPhidt = omega_array \
        + sigma*(1+R**2)/(2*N*R)*(np.cos(Phi)*np.dot(redA, R*np.sin(Phi))
                                  - np.sin(Phi)*np.dot(redA, R*np.cos(Phi)))
    return np.concatenate([dRdt, dPhidt])


def reduced_kuramoto_sakaguchi_old(t, Z, kappa, hatkappa, omega, sigma, alpha, N):
    """
    Older function then reduced_kuramoto_sakaguchi_2, I don't use kappa
    and hatkappa anymore in the threefold dimension reduction
    :param t:
    :param Z:
    :param kappa:
    :param hatkappa:
    :param omega:
    :param sigma:
    :param alpha:
    :param N:
    :return:
    """
    # \
    # - sigma/N*(hatkappa-kappa)*np.conj(Z)*Z**2*np.exp(1j*alpha)
    return 1j*omega*Z + 0.5*sigma*kappa/N*(Z*np.exp(-1j*alpha)
                                           - np.conj(Z)*Z**2*np.exp(1j*alpha))


def reduced_kuramoto_sakaguchi(t, Z, reduced_A, reduced_K, reduced_W,
                               coupling, alpha, N):
    """
    Function to integrate the Sakaguchi-Kuramoto dynamics with ode.
    The coupling term is given by coupling/N. In the formalism of the
    threefold dimension reduction

    :param w:
    :param t:
    :param reduced_A: With no D_alpha
    :param reduced_K: With D_alpha
    :param reduced_W:
    :param alpha:
    :return:

    The shape of these matrix is (q, numberoftimepoints).
    q is the reduced dimension (number of blocks)
    """
    return 1j*np.dot(reduced_W, Z) \
        + 0.5*coupling/N*np.exp(-1j*alpha)*np.dot(reduced_A, Z) \
        - 0.5*coupling/N*np.exp(1j*alpha)*np.conj(np.dot(reduced_A, Z))\
        * (np.dot(reduced_K, Z))**2


def reduced_kuramoto(t, Z, redA, omega, sigma, N, redK, hatredA, hatredK):
    return 1j*omega*Z + 0.5*sigma/N*np.dot(redA, Z) \
           - 0.5*sigma/N*Z**2*np.dot(redA, np.conj(Z)) \
           - sigma/N*Z**2*np.dot(hatredK - redK, np.conj(Z)) \
           + 0.5*sigma/N*np.dot(hatredA - redA, Z) \
           - 0.5*sigma/N*Z**2*np.dot(hatredA - redA, Z*np.conj(Z)**2)


def reduced_kuramoto_degree_freq(t, Z, kappa, hatkappa, sigma, N):
    return 1j*hatkappa*Z + 0.5*sigma*kappa/N*Z \
           - sigma*(hatkappa - kappa)/N*np.conj(Z)*Z**2


def reduced_kuramoto_Rwp(t, RwP, kappa, primekappa,
                         D, redomega, primeomega, sigma, N):
    R, w, Psi = RwP

    dRdt = w*np.sin(Psi) + 0.5*kappa*sigma/N*R*(1 - R**2)

    dwdt = -D*R*np.sin(Psi) - sigma/N*primekappa*R**2*w

    dPsidt = redomega - primeomega + 1 + (w/R - D*R/w)*np.cos(Psi)

    return np.array([dRdt, dwdt, dPsidt])


def reduced_kuramoto_freq_1D(t, ZW, kappa, hatkappa, primekappa,
                             var_omega, redomega, tau, eta, sigma, N):
    Z, W = ZW

    dZdt = 1j*redomega*Z + 1j*W \
        + 0.5*sigma/N*(kappa*Z - (2*hatkappa - kappa)*np.conj(Z)*Z**2)
    dWdt = 1j*var_omega*Z - 1j*redomega*W \
        + 0.5*sigma/N*((tau - kappa*redomega)*(Z - np.conj(Z)*Z**2)
                       + eta*(1 - Z**2)*W - 2*primekappa*np.conj(Z)*Z*W)

    return np.array([dZdt, dWdt])


def reduced_kuramoto_freq_1D_2(t, ZW, kappa, hatkappa, primekappa_list,
                               omega_moments, tau_list, eta_list, sigma, N):
    Z, W, W2 = ZW

    omega1, omega2, omega3 = omega_moments
    tau1, tau2 = tau_list
    eta1, eta2 = eta_list
    primekappa1, primekappa2 = primekappa_list

    dZdt = 1j*omega1*Z + 1j*W \
        + 0.5*sigma/N*(kappa*Z - (2*hatkappa - kappa)*np.conj(Z)*Z**2)

    dWdt = 1j*(omega2 - omega1**2)*Z + 1j*(W2 - omega1*W) \
        + 0.5*sigma/N*((tau1 - kappa*omega1)*(Z - np.conj(Z)*Z**2)
                       + eta1*(1 - Z**2)*W
                       - 2*primekappa1*np.conj(Z)*Z*W)

    dW2dt = 1j*(omega3 - omega1*omega2)*Z - 1j*omega2*W \
            + 0.5*sigma/N*((tau2 - kappa*omega2)*(Z - np.conj(Z)*Z**2)
                           + eta2*(1 - Z**2)*W2
                           - 2*primekappa2*np.conj(Z)*Z*W2)

    return np.array([dZdt, dWdt, dW2dt])


def reduced_kuramoto_old(t, Z, redA, omega, sigma, N, kappa, redK):
    return 1j * np.dot(omega, Z) \
           + 0.5 * sigma * kappa / N * (kappa ** (-1) * np.dot(redA, Z)
                                        - kappa ** (-1) * np.conj(
                np.dot(redA, Z))
                                        * (kappa ** (-1) * np.dot(redK,
                                                                  Z)) ** 2)


def reduced_kuramoto_freq(t, ZW, redA, sigma, N,
                          redomega, redomega2, redOmega,
                          hatredOmega, hatredK):
    q = int(len(ZW) / 2)
    Z, W = ZW[0:q], ZW[q:]
    dZdt = 1j * W + 1j * redomega * Z + 0.5 * sigma / N * np.dot(redA, Z) \
           - 0.5 * sigma / N * Z ** 2 * np.dot(redA, np.conj(Z))
    dWdt = 1j * (redomega2 - redomega ** 2) * Z - 1j * redomega * W \
           + 0.5 * sigma / N * np.dot(redOmega, Z) \
           - sigma / N * Z * W * np.dot(hatredK, np.conj(Z)) \
           + 0.5 * sigma / N * np.dot(hatredOmega - redOmega - redomega * redA,
                                      Z) \
           - 0.5 * sigma / N * Z ** 2 * np.dot(
        hatredOmega - redOmega - redomega * redA,
        np.conj(Z))
    return np.concatenate([dZdt, dWdt])


def reduced_kuramoto_freq_old(t, ZW, redA, sigma, N, redomega, redomega2):
    q = int(len(ZW) / 2)
    Z, W = ZW[0:q], ZW[q:]
    dZdt = 1j * W + 1j * redomega * Z + 0.5 * sigma / N * np.dot(redA, Z) \
           - 0.5 * sigma / N * Z ** 2 * np.dot(redA, np.conj(Z))
    dWdt = 1j * (redomega2 - redomega ** 2) * Z - 1j * redomega * W \
           - sigma / N * Z * W * np.dot(redA, np.conj(Z))
    return np.concatenate([dZdt, dWdt])


# def reduced_kuramoto_2D(t, Z, redA, omega_array, sigma, N, hatredA, hatLambda):
#     return 1j*omega_array*Z + 0.5*sigma/N*np.dot(redA, Z) \
#            - 0.5*sigma/N*Z**2*np.dot(redA, np.conj(Z)) \
#            #+ sigma/N*Z**2*np.dot(hatredA - redA, np.conj(Z)) \
#            #- 0.5*sigma/N*Z**2*np.dot(hatLambda - redA, Z*np.conj(Z)**2)
#
#
# def reduced_kuramoto_freq_1D(t, ZW, kappa, hatkappa, primekappa,
#                             var_omega, omega, hatomega, primeomega, sigma, N):
#    Z, W = ZW
#                                  # (2*hatkappa/kappa - 1)*
#    dZdt = 1j*omega*Z + 1j*W \
#        + 0.5*sigma*kappa/N*(Z - np.conj(Z)*Z**2)
#    dWdt = 1j*var_omega*Z + 1j*(primeomega - 1)*W\
#        - sigma/N*primekappa*np.conj(Z)*Z*W \
#        #+ 0.5*sigma*kappa/N*((hatomega/omega - 1)*Z
#        #                     - (hatomega/omega
#        #                        - 2*hatkappa/kappa + 1)*np.conj(Z)*Z**2)
#    return np.array([dZdt, dWdt])
#
# def reduced_kuramoto_freq_1D(t, ZW, kappa, hatkappa, primekappa,
#                             D, omega, hatomega, primeomega, sigma, N):
#    Z, W = ZW
#                                  # (2*hatkappa/kappa - 1)*
#    dZdt = 1j*omega*Z + 1j*omega*W \
#        + 0.5*sigma*kappa/N*(Z - np.conj(Z)*Z**2)
#    dWdt = 1j*D*Z + 1j*(primeomega - omega)*W\
#        - sigma/N*primekappa*np.conj(Z)*Z*W \
#        #+ 0.5*sigma*kappa/N*((hatomega/omega - 1)*Z
#        #                     - (hatomega/omega
#        #                        - 2*hatkappa/kappa + 1)*np.conj(Z)*Z**2)
#    return np.array([dZdt, dWdt])
''''''


# Lorenz
@jit(nopython=True)
def reduced_lorenz(t, X, MAMp, MWMp, MKMp, sigma, Omega, a, b, c):
    n = len(MAMp[0])
    x, y, z = X[0:n], X[n: 2*n], X[2*n:3*n]
    redL = MKMp - MAMp
    # print(x, y, z)
    dxdt = MWMp@(a*(y - x)) - sigma*redL@x
    dydt = MWMp@(b*x - y) - (MWMp@x)*(MWMp@z)/Omega
    dzdt = (MWMp@x)*(MWMp@y)/Omega - c*MWMp@z
    return np.concatenate((dxdt, dydt, dzdt))


# Rossler
@jit(nopython=True)
def reduced_rossler(t, X, MAMp, MWMp, MKMp, sigma, Omega, a, b, c):
    n = len(MAMp[0])
    x, y, z = X[0:n], X[n: 2*n], X[2*n:3*n]
    redL = MKMp - MAMp
    dxdt = MWMp@(-y - z) - sigma*redL@x
    dydt = MWMp@(x + a*y)
    dzdt = b*Omega - c*MWMp@z + (MWMp@x)*(MWMp@z)/Omega
    return np.concatenate((dxdt, dydt, dzdt))


# Wilson-Cowan
def reduced_cowan_wilson(t, W, kappa, tau, mu, hatkappa,
                         kappa2, gamma, epsilon, tauL, tauR):
    # Nonlinear reduction
    X, varX, Cxx = W

    exp = np.exp(-tau*(X-mu))
    sig = 1/(1+np.exp(-tau*(X-mu)))

    dXdt = -X + kappa*sig \
           - 0.5*kappa*tau**2*varX*exp*sig**2*(1 - 2*exp*sig)
    dvarXdt = -2*varX + 2*(hatkappa - kappa)*X*sig + 2*tau*Cxx*exp*sig**2
    dCxxdt = -2*Cxx + (epsilon - kappa2 + gamma - kappa**2)*X*sig\
        + tau*(tauL+tauR)*Cxx*exp*sig**2 - (hatkappa - kappa)*X*(-X
        + kappa*sig - 0.5*kappa*tau**2*varX*exp*sig**2*(1 - 2*exp*sig))
    return np.array([dXdt, dvarXdt, dCxxdt])


def reduced_cowan_wilson_2D(t, W, redA, tau, mu, hatLambda, redK):
    # Nonlinear reduction,
    X, varX = W[0:2], W[2:4]
    # C0, C1 = W[4:6], W[6:8]

    exp = np.exp(-tau*(X-mu))
    sig = 1/(1+np.exp(-tau*(X-mu)))
    # C_crossed = np.array([C0, C1])

    dXdt = -X + np.dot(redA, sig) \
           - 0.5*tau**2*np.dot(hatLambda, varX*exp*sig**2*(1 - 2*exp*sig))
    dvarXdt = -2*varX + 2*X*np.dot(redK-redA, sig)# \
    # + 2*tau*np.dot(C_crossed, exp*sig**2)
    # dC0dt = -2*C0 + X[0]*(redK[0] - redA[0])*(np.dot(redA, sig)
    #        - 0.5*tau**2*np.dot(hatLambda, varX*exp*sig**2*(1 - 2*exp*sig)))
    # dC1dt = -2*C1 + X[0]*(redK[0] - redA[0])*(np.dot(redA, sig)
    #            - 0.5*tau**2*np.dot(hatLambda, varX*exp*sig**2*(1-2*exp*sig)))

    return np.concatenate([dXdt, dvarXdt])  # , dC0dt, dC1dt])


def reduced_cowan_wilson_DART(t, x, redW, tau, mu, kappa):
    return -x + kappa*(1/(1+np.exp(-tau*((redW@x/kappa)-mu))))


def reduced_wilson_cowan_factorization(t, x, L, M, a, b):
    return -x + M@(1/(1+np.exp(-a*(L@x - b))))


def reduced_wilson_cowan_XR(t, x, W, P, a, b):
    return -x + P@(1/(1+np.exp(-a*(W@x - b))))


def reduced_wilson_cowan_XL_autonomous_part(t, x, W, P, a, b):
    return -x


def reduced_wilson_cowan_hebb(t, X, redD, Q, q, gamma, a, b, c):
    Dflat = redD.flatten()
    n = len(redD[0])
    x, w = X[0:n], X[n: n+n**2]
    y = np.reshape(w, (n, n))@(q*x) + gamma
    z = q**(-1)*(x.T@Q).T
    dxdt = -x + 1/(1+np.exp(-a*(y - b)))
    dwdt = Dflat*np.tile(z, n)*np.repeat(x, n) - c*w
    return np.concatenate((dxdt, dwdt))


def reduced_wilson_cowan_oja(t, X, redD, Q, q, gamma, a, b, c):
    Dflat = redD.flatten()
    n = len(redD[0])
    x, w = X[0:n], X[n: n+n**2]
    y = np.reshape(w, (n, n))@(q*x) + gamma
    z = q**(-1)*(x.T@Q).T
    dxdt = -x + 1/(1+np.exp(-a*(y - b)))
    dwdt = Dflat*np.tile(z, n)*np.repeat(x, n) - c*w*np.repeat(x**2, n)
    return np.concatenate((dxdt, dwdt))


def reduced_wilson_cowan_BCM(t, X, redD, Q, q, tautx, gamma, a, b, c, xmax):
    """

    :param t:
    :param X:
    :param redD:
    :param Q:
    :param q = np.sum(Q.T, axis=1)
    :param tau:
    :param tauw:
    :param taut:
    :param a:
    :param b:
    :param c:
    :param xmax:
    :return:
    """
    Dflat = redD.flatten()
    n = len(redD[0])
    x, w, theta = X[0:n], X[n: n+n**2], X[n+n**2:2*n+n**2]
    y = np.reshape(w, (n, n))@(q*x) + gamma
    z = q**(-1)*(x.T@Q).T
    dxdt = -x + 1/(1+np.exp(-a*(y - b)))
    dwdt = Dflat*np.tile(z, n)*np.repeat(x*(x-theta), n) - c*w
    dthetadt = (x**2 - theta*xmax)/tautx
    return np.concatenate((dxdt, dwdt, dthetadt))


def reduced_wilson_cowan_factorization_BCM(t, X, redD, M, Mp, Q, p, q,
                                           tautx, a, b, c, xmax):
    """

    :param t:
    :param X:
    :param redD:
    :param Q:
    :param q = np.sum(Q.T, axis=1)
    :param tau:
    :param tauw:
    :param taut:
    :param a:
    :param b:
    :param c:
    :param xmax:
    :return:
    """
    Dflat = redD.flatten()
    n = len(redD[0])
    x, w, theta = X[0:n], X[n: n+n**2], X[n+n**2:2*n+n**2]
    z = q**(-1)*(x.T@Q).T
    redW = np.reshape(w, (n, n))
    dxdt = -x + M@(1/(1+np.exp(-a*(Mp@redW@x - b))))
    dwdt = Dflat*np.tile(z, n)*np.repeat(x*(x-theta)/p**2, n) - c*w
    dthetadt = (x**2/p - theta*xmax)/tautx
    return np.concatenate((dxdt, dwdt, dthetadt))
# l'équation pour l'activité est complètement indépendante de l'équation de w et theta !

# Hopf
def reduced_hopf(t, ZW, K, meanomega, varomega):
    Z, W = ZW
    dZdt = 1j*W + (1 - np.absolute(Z)**2 + 1j*meanomega)*Z
    dWdt = 1j*varomega*Z \
        + (1 - K - 2*np.absolute(Z)**2 - 1j*meanomega)*W - Z**2*np.conj(W)
    # In the article De Monte 2002, here^, it is a + sign, but I think it is an
    # error.
    return np.array([dZdt, dWdt])


# Cosinus
def reduced_cosinus_2D(t, Z, redA, omega_array, sigma, N, hatredA, hatLambda):
    return 1j*omega_array*Z + 0.5*1j*sigma/N*np.dot(redA, Z + np.conj(Z))


# def reduced_hopf(t, ZW, K, meanomega, varomega):
#    Z, W = ZW
#    dZdt = 1j*W + (1 - np.absolute(Z)**2 + 1j*meanomega)*Z
#    dWdt = 1j*varomega*Z \
#        + (1 - K - 2*np.absolute(Z)**2 - 1j*meanomega)*W - Z**2*np.conj(W)
#    # In the article De Monte 2002, here^, it is a + sign, but I think it
#    # is an error.
#    return np.array([dZdt, dWdt])
