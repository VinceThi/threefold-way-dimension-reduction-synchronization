from synch_predictions.dynamics.integrate import *
from synch_predictions.dynamics.dynamics import *
from synch_predictions.dynamics.reduced_dynamics import *
# from synch_predictions.graphs.graph_spectrum import *
import numpy as np
from numpy.linalg import multi_dot # , pinv
import networkx as nx
import time
from tqdm import tqdm


def plot_RPhi_vs_Z_winfree_2D(p_out_array, p_in, sizes, nb_instances,
                              t0, t1, dt, omega, omega_array,
                              sigma, N, mean_SBM=False):
    """
    Predictions for the Winfree dynamics on modular graphs when one module has
    one natural frequency and the other has another natural frequency

    :param p_out_array: Array of probabilities, for a node, to be connected
                        outside its community.
    :param p_in: Probability, for a node, to be connected inside its community.
                 If p_in = 0, it is a SBM graph.
    :param sizes: [n1, n2] Community sizes
    :param nb_instances: The number of instances of graphs
                         and initial conditions
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param omega: 1xN : n1*[omega1] + n2*[omega2]
    :param omega_array: 1x2 : [omega1, omega2]
    :param sigma: Coupling constant
    :param N: Number of nodes
    :param mean_SBM: (Boolean)If we want to get the transition for the mean
                     stochastic block model, it is True.
    :param plot_temporal_series:

    :return: Matrices nb_instances x len(p_out_array) measuring the phase
             synchronization in each community and the global synchronization
    """

    n1, n2 = sizes
    for p_out in tqdm(p_out_array):
        for instance in range(nb_instances):
            time.sleep(3)

            pq = [[p_in, p_out], [p_out, p_in]]

            M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                            [np.zeros(n1), 1 / n2 * np.ones(n2)]])
            P = (np.block([[np.ones(n1), np.zeros(n2)],
                           [np.zeros(n1), np.ones(n2)]])).T

            if mean_SBM:
                A = np.zeros((N, N))
                ii = 0
                for i in range(0, len(sizes)):
                    jj = 0
                    for j in range(0, len(sizes)):
                        A[ii:ii + sizes[i], jj:jj + sizes[j]] \
                            = pq[i][j] * np.ones((sizes[i], sizes[j]))
                        jj += sizes[j]
                    ii += sizes[i]

                M = M_0

            else:
                A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
                if p_in > 0:
                    M = M_0
                else:
                    # V = get_eigenvectors_matrix(A, 2)  # Not normalized
                    # Vp = pinv(V)
                    # C = np.dot(M_0, Vp)
                    # CV = np.dot(C, V)
                    M = M_0  # (CV.T / (np.sum(CV, axis=1))).T  #

            # Reduced paramters
            redA = multi_dot([M, A, P])
            # hatredA = (multi_dot([M ** 2, A, P]).T /
            #            np.diag(np.dot(M, M.T))).T
            # hatLambda = multi_dot([M, A, pinv(M)])

            theta0 = 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            Z0 = np.dot(M, z0)
            W0 = np.concatenate([np.absolute(Z0), np.angle(Z0)])

            # Integrate complete dynamics
            args_winfree = (omega, sigma)
            winfree_sol = integrate_dynamics(t0, t1, dt, winfree, A,
                                             "vode", theta0,
                                             *args_winfree)

            r1 = np.absolute(
                np.sum(M[0, 0:n1] * np.exp(1j * winfree_sol[:, 0:n1]),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, n1:] * np.exp(1j * winfree_sol[:, n1:]),
                       axis=1))

            # Integrate complex reduced dynamics
            args_red_winfree_2D_Z = (omega_array, sigma, N,
                                     redA, redA)
            red_winfree_sol_Z = integrate_dynamics(t0, t1, dt,
                                                   reduced_winfree_2D,
                                                   redA, "zvode", Z0,
                                                   *args_red_winfree_2D_Z)

            Z1, Z2 = red_winfree_sol_Z[:, 0], red_winfree_sol_Z[:, 1]
            R1_Z, R2_Z = np.absolute(Z1), np.absolute(Z2)
            Phi1_Z, Phi2_Z = np.angle(Z1), np.angle(Z2)

            # Integrate real reduced dynamics
            args_red_winfree_2D = (omega_array, sigma, N)
            red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_winfree_R_Phi_2D,
                                                 redA, "vode", W0,
                                                 *args_red_winfree_2D)

            R1, R2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
            Phi1, Phi2 = red_winfree_sol[:, 2], red_winfree_sol[:, 3]

            import matplotlib.pyplot as plt

            plt.subplot(211)
            plt.plot(R1_Z, color="b")
            plt.plot(R1, color="lightblue")
            # plt.plot(R1_Z*np.cos(Phi1_Z), R1_Z*np.sin(Phi1_Z), color="b")
            # plt.plot(R1*np.cos(Phi1), R1*np.sin(Phi1), color="lightblue")
            plt.ylabel("$R_1$", fontsize=12)
            # plt.ylim([-0.02, 1.02])

            plt.subplot(212)
            plt.plot(R2_Z, color="r")
            plt.plot(R2, color="y")
            # plt.plot(R2_Z*np.cos(Phi2_Z), R2_Z*np.sin(Phi2_Z), color="r")
            # plt.plot(R2*np.cos(Phi2), R2*np.sin(Phi2), color="y")
            plt.ylabel("$R_2$", fontsize=12)
            # plt.ylim([-0.02, 1.02])
            plt.xlabel("$t$", fontsize=12)
            plt.tight_layout()

            plt.show()


def plot_RPhi_vs_Z_kuramoto_2D(p_out_array, p_in, sizes, nb_instances,
                              t0, t1, dt, omega, omega_array,
                              sigma, N, mean_SBM=False):
    """
    Predictions for the Kuramoto dynamics on modular graphs when one module has
    one natural frequency and the other has another natural frequency

    :param p_out_array: Array of probabilities, for a node, to be connected
                        outside its community.
    :param p_in: Probability, for a node, to be connected inside its community.
                 If p_in = 0, it is a SBM graph.
    :param sizes: [n1, n2] Community sizes
    :param nb_instances: The number of instances of graphs
                         and initial conditions
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param omega: 1xN : n1*[omega1] + n2*[omega2]
    :param omega_array: 1x2 : [omega1, omega2]
    :param sigma: Coupling constant
    :param N: Number of nodes
    :param mean_SBM: (Boolean)If we want to get the transition for the mean
                     stochastic block model, it is True.
    :param plot_temporal_series:

    :return: Matrices nb_instances x len(p_out_array) measuring the phase
             synchronization in each community and the global synchronization
    """

    n1, n2 = sizes
    for p_out in tqdm(p_out_array):
        for instance in range(nb_instances):
            time.sleep(3)

            pq = [[p_in, p_out], [p_out, p_in]]

            M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                            [np.zeros(n1), 1 / n2 * np.ones(n2)]])
            P = (np.block([[np.ones(n1), np.zeros(n2)],
                           [np.zeros(n1), np.ones(n2)]])).T

            if mean_SBM:
                A = np.zeros((N, N))
                ii = 0
                for i in range(0, len(sizes)):
                    jj = 0
                    for j in range(0, len(sizes)):
                        A[ii:ii + sizes[i], jj:jj + sizes[j]] \
                            = pq[i][j] * np.ones((sizes[i], sizes[j]))
                        jj += sizes[j]
                    ii += sizes[i]

                M = M_0

            else:
                A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
                if p_in > 0:
                    M = M_0
                else:
                    # V = get_eigenvectors_matrix(A, 2)  # Not normalized
                    # Vp = pinv(V)
                    # C = np.dot(M_0, Vp)
                    # CV = np.dot(C, V)
                    M = M_0  # (CV.T / (np.sum(CV, axis=1))).T

            # Reduced paramters
            redA = multi_dot([M, A, P])
            # hatredA = (multi_dot([M ** 2, A, P]).T /
            #            np.diag(np.dot(M, M.T))).T
            # hatLambda = multi_dot([M, A, pinv(M)])

            theta0 = 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            Z0 = np.dot(M, z0)
            W0 = np.concatenate([np.absolute(Z0), np.angle(Z0)])

            # Integrate complete dynamics
            args_kuramoto = (omega, sigma)
            kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                              "vode", theta0,
                                              *args_kuramoto)

            r1 = np.absolute(
                np.sum(M[0, 0:n1] * np.exp(1j * kuramoto_sol[:, 0:n1]),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, n1:] * np.exp(1j * kuramoto_sol[:, n1:]),
                       axis=1))

            # Integrate complex reduced dynamics
            args_red_kuramoto_2D_Z = (omega_array, sigma, N,
                                      redA, redA)
            red_kuramoto_sol_Z = integrate_dynamics(t0, t1, dt,
                                                    reduced_kuramoto_2D,
                                                    redA, "zvode", Z0,
                                                    *args_red_kuramoto_2D_Z)

            Z1, Z2 = red_kuramoto_sol_Z[:, 0], red_kuramoto_sol_Z[:, 1]
            R1_Z, R2_Z = np.absolute(Z1), np.absolute(Z2)
            Phi1_Z, Phi2_Z = np.angle(Z1), np.angle(Z2)

            # Integrate real reduced dynamics
            args_red_kuramoto_2D = (omega_array, sigma, N)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_R_Phi_2D,
                                                  redA, "vode", W0,
                                                  *args_red_kuramoto_2D)

            R1, R2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            Phi1, Phi2 = red_kuramoto_sol[:, 2], red_kuramoto_sol[:, 3]

            import matplotlib.pyplot as plt

            plt.subplot(211)
            plt.plot(r1, color="k")
            plt.plot(R1_Z, color="b")
            plt.plot(R1, color="lightblue")
            # plt.plot(R1_Z*np.cos(Phi1_Z), R1_Z*np.sin(Phi1_Z), color="b")
            # plt.plot(R1*np.cos(Phi1), R1*np.sin(Phi1), color="lightblue")
            plt.ylabel("$R_1$", fontsize=12)
            plt.ylim([-0.02, 1.02])

            plt.subplot(212)
            plt.plot(r2, color="k")
            plt.plot(R2_Z, color="r")
            plt.plot(R2, color="y")
            # plt.plot(R2_Z*np.cos(Phi2_Z), R2_Z*np.sin(Phi2_Z), color="r")
            # plt.plot(R2*np.cos(Phi2), R2*np.sin(Phi2), color="y")
            plt.ylabel("$R_2$", fontsize=12)
            plt.ylim([-0.02, 1.02])
            plt.xlabel("$t$", fontsize=12)
            plt.tight_layout()

            plt.show()


def plot_RPhi_vs_Z_theta_2D(p_out_array, p_in, sizes, nb_instances,
                            t0, t1, dt, I, I_array,
                            sigma, N, mean_SBM=False):
    """
    Predictions for the theta dynamics on modular graphs

    :param p_out_array: Array of probabilities, for a node, to be connected
                        outside its community.
    :param p_in: Probability, for a node, to be connected inside its community.
                 If p_in = 0, it is a SBM graph.
    :param sizes: [n1, n2] Community sizes
    :param nb_instances: The number of instances of graphs
                         and initial conditions
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param I: 1xN : n1*[I1] + n2*[I2]
    :param I_array: 1x2 : [I1, I2]
    :param sigma: Coupling constant
    :param N: Number of nodes
    :param mean_SBM: (Boolean)If we want to get the transition for the mean
                     stochastic block model, it is True.

    :return: Matrices nb_instances x len(p_out_array) measuring the phase
             synchronization in each community and the global synchronization
    """

    n1, n2 = sizes
    for p_out in tqdm(p_out_array):
        for instance in range(nb_instances):
            time.sleep(3)

            pq = [[p_in, p_out], [p_out, p_in]]

            M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                            [np.zeros(n1), 1 / n2 * np.ones(n2)]])
            P = (np.block([[np.ones(n1), np.zeros(n2)],
                           [np.zeros(n1), np.ones(n2)]])).T

            if mean_SBM:
                A = np.zeros((N, N))
                ii = 0
                for i in range(0, len(sizes)):
                    jj = 0
                    for j in range(0, len(sizes)):
                        A[ii:ii + sizes[i], jj:jj + sizes[j]] \
                            = pq[i][j] * np.ones((sizes[i], sizes[j]))
                        jj += sizes[j]
                    ii += sizes[i]

                M = M_0

            else:
                A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
                if p_in > 0:
                    M = M_0
                else:
                    # V = get_eigenvectors_matrix(A, 2)  # Not normalized
                    # Vp = pinv(V)
                    # C = np.dot(M_0, Vp)
                    # CV = np.dot(C, V)
                    M = M_0  # (CV.T / (np.sum(CV, axis=1))).T

            # Reduced paramters
            redA = multi_dot([M, A, P])
            # hatredA = (multi_dot([M ** 2, A, P]).T /
            #            np.diag(np.dot(M, M.T))).T
            # hatLambda = multi_dot([M, A, pinv(M)])

            theta0 = 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            Z0 = np.dot(M, z0)
            W0 = np.concatenate([np.absolute(Z0), np.angle(Z0)])

            # Integrate complete dynamics
            args_theta = (I, sigma)
            theta_sol = integrate_dynamics(t0, t1, dt, theta, A,
                                           "vode", theta0,
                                           *args_theta)

            r1 = np.absolute(
                np.sum(M[0, 0:n1] * np.exp(1j * theta_sol[:, 0:n1]),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, n1:] * np.exp(1j * theta_sol[:, n1:]),
                       axis=1))

            # Integrate complex reduced dynamics
            args_red_theta_2D_Z = (I_array, sigma, N,
                                   redA, redA)
            red_theta_sol_Z = integrate_dynamics(t0, t1, dt,
                                                 reduced_theta_2D,
                                                 redA, "zvode", Z0,
                                                 *args_red_theta_2D_Z)

            Z1, Z2 = red_theta_sol_Z[:, 0], red_theta_sol_Z[:, 1]
            R1_Z, R2_Z = np.absolute(Z1), np.absolute(Z2)
            Phi1_Z, Phi2_Z = np.angle(Z1), np.angle(Z2)

            # Integrate real reduced dynamics
            args_red_theta_2D = (I_array, sigma, N)
            red_theta_sol = integrate_dynamics(t0, t1, dt,
                                               reduced_theta_R_Phi_2D,
                                               redA, "vode", W0,
                                               *args_red_theta_2D)

            R1, R2 = red_theta_sol[:, 0], red_theta_sol[:, 1]
            Phi1, Phi2 = red_theta_sol[:, 2], red_theta_sol[:, 3]

            import matplotlib.pyplot as plt

            plt.subplot(211)
            plt.plot(r1, color="k")
            plt.plot(R1_Z, color="b")
            plt.plot(R1, color="lightblue")
            # plt.plot(R1_Z*np.cos(Phi1_Z), R1_Z*np.sin(Phi1_Z), color="b")
            # plt.plot(R1*np.cos(Phi1), R1*np.sin(Phi1), color="lightblue")
            plt.ylabel("$R_1$", fontsize=12)
            plt.ylim([-0.02, 1.02])

            plt.subplot(212)
            plt.plot(r2, color="k")
            plt.plot(R2_Z, color="r")
            plt.plot(R2, color="y")
            # plt.plot(R2_Z*np.cos(Phi2_Z), R2_Z*np.sin(Phi2_Z), color="r")
            # plt.plot(R2*np.cos(Phi2), R2*np.sin(Phi2), color="y")
            plt.ylabel("$R_2$", fontsize=12)
            plt.ylim([-0.02, 1.02])
            plt.xlabel("$t$", fontsize=12)
            plt.tight_layout()

            plt.show()


if __name__ == "__main__":
    # Winfree
    """
    # Time parameters
    t0, t1, dt = 0, 100, 0.05   # 1000, 0.05
    time_list = np.linspace(t0, t1, int(t1 / dt))
    sizes = [150, 100]
    n1, n2 = sizes
    N = sum(sizes)
    p_in = 0
    p_out_array = np.linspace(1/N, 1, 10)
    nb_instances = 1
    sigma = 1
    omega1 = 0.3  # -np.pi/2
    omega2 = -n1/n2 * omega1
    omega = np.concatenate([omega1 * np.ones(n1), omega2 * np.ones(n2)])
    omega_array = np.array([omega1, omega2])

    plot_RPhi_vs_Z_winfree_2D(p_out_array, p_in, sizes, nb_instances,
                              t0, t1, dt, omega, omega_array,
                              sigma, N, mean_SBM=False)
    """

    # Kuramoto
    """
    # Time parameters                                                    
    t0, t1, dt = 0, 1000, 0.05   # 1000, 0.05                             
    time_list = np.linspace(t0, t1, int(t1 / dt))                        
    sizes = [150, 100]                                                   
    n1, n2 = sizes                                                       
    N = sum(sizes)                                                       
    p_in = 0                                                             
    p_out_array = np.linspace(1/N, 1, 10)                                
    nb_instances = 1                                                     
    sigma = 1                                                            
    omega1 = 0.1                                            
    omega2 = -n1/n2 * omega1                                             
    omega = np.concatenate([omega1 * np.ones(n1), omega2 * np.ones(n2)]) 
    omega_array = np.array([omega1, omega2])                             

    plot_RPhi_vs_Z_kuramoto_2D(p_out_array, p_in, sizes, nb_instances,    
                               t0, t1, dt, omega, omega_array,            
                               sigma, N, mean_SBM=False)                  
    """

    # Theta
    # """
    # Time parameters
    t0, t1, dt = 0, 100, 0.001
    time_list = np.linspace(t0, t1, int(t1 / dt))
    sizes = [150, 100]
    n1, n2 = sizes
    N = sum(sizes)
    p_in = 0.9
    p_out_array = np.linspace(0.5, 1, 10)
    nb_instances = 1
    sigma = 4
    I1 = -1
    I2 = -1
    I = np.concatenate([I1 * np.ones(n1), I2 * np.ones(n2)])
    I_array = np.array([I1, I2])

    plot_RPhi_vs_Z_theta_2D(p_out_array, p_in, sizes, nb_instances,
                            t0, t1, dt, I, I_array,
                            sigma, N, mean_SBM=False)
    # """
