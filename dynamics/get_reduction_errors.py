from dynamics.integrate import *
from dynamics.dynamics import *
from dynamics.reduced_dynamics import *
from graphs.graph_spectrum import *
import numpy as np
from numpy.linalg import multi_dot, pinv
import networkx as nx
import time
from tqdm import tqdm


def RMSE(a, b):
    return np.sqrt(np.mean((a - b)**2))


def L1(a, b):
    return np.abs(a - b)


def mean_L1(a, b):
    return np.mean(L1(a, b))


def get_error_vs_N_kuramoto_bipartite(R_dictionary, p_out_array, omega1_array,
                                      N_array, plot_transitions):
    # L1_freq_vs_N = []
    L1_spec_vs_N = []
    # var_L1_freq_vs_N = []
    var_L1_spec_vs_N = []

    for N in tqdm(N_array):
      
        r_array = np.array(R_dictionary["r{}".format(N)])
        R_array = np.array(R_dictionary["R{}".format(N)])
        # r_uni_array = np.array(R_dictionary["r_uni{}".format(N)])
        # R_uni_array = np.array(R_dictionary["R_uni{}".format(N)])

        # L1_freq_vs_N.append(np.mean(L1(r_uni_array, R_uni_array)))
        L1_spec_vs_N.append(np.mean(L1(r_array, R_array)))

        # var_L1_freq_vs_N.append(np.var(L1(r_uni_array, R_uni_array)))
        var_L1_spec_vs_N.append(np.var(L1(r_array, R_array)))

        if plot_transitions:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(3, 3))
            y_limits_L1 = None  # [0, 0.5]
            ax1 = plt.subplot(1, 1, 1)
            # See R vs p_out
            plt.plot(p_out_array, r_array[:, 2], color="#252525",
                     label="$r_{spec}$")
            plt.plot(p_out_array, R_array[:, 2], color="#969696",
                     linestyle="--",
                     label="$R_{spec}$")
            # plt.plot(p_out_array, r_uni_array[:, 2], color="#9e9ac8",
            #          label="$r_{uni}$")
            # plt.plot(p_out_array, R_uni_array[:, 2], color="#cbc9e2",
            #          linestyle="--",
            #          label="$R_{uni}$")

            # See R vs omega1
            # plt.plot(p_out_array, r_array[10, :], color="#252525",
            #          label="$r_{spec}$")
            # plt.plot(p_out_array, R_array[10, :], color="#969696",
            #          linestyle="--",
            #          label="$R_{spec}$")
            # plt.plot(p_out_array, r_uni_array[10, :], color="#9e9ac8",
            #          label="$r_{uni}$")
            # plt.plot(p_out_array, R_uni_array[10, :], color="#cbc9e2",
            #          linestyle="--",
            #          label="$R_{uni}$")

            plt.ylim([0, 1.05])
            plt.ylabel("$R$", fontsize=12, labelpad=5)
            plt.xlabel("$p_{out}$", fontsize=12)
            plt.legend(loc=4, fontsize=10)
            plt.tight_layout()

            plt.show()

    return L1_spec_vs_N, var_L1_spec_vs_N


def get_data_kuramoto_one_star(sigma_array, N, t0, t1, dt, t0_red, t1_red,
                               dt_red, averaging, plot_temporal_series=0):
    """
    Generate data for different reduced Kuramoto dynamics on a star graph

    IMPORTANT: The initial conditions of the complete and the reduced dynamics
    are different to simplify the convergence to equilibrium.

    See Gomez-Gardenes 2011 to know how to integrate the system and get the
    complete hysteresis.

    :param sigma_array: Array of the coupling constant sigma
    :param N: Number of nodes
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param t0_red: Initial time (reduced dynamics)
    :param t1_red: Final time (reduced dynamics)
    :param dt_red: Time step (reduced dynamics)
    :param averaging: number between 5 and 10. ex: 8
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [--- r ---],
                               "r1",            [--- r1---],
                               "r2",            [--- r2---],
                               "R",             [--- R ---],
                               "R1",            [--- R1---],
                               "R2",            [--- R2---],
                               sigma_array,    sigma_array}
            where the values [---X---] is an array of shape
            1 times len(sigma_list) of the order parameter X.
            R is the spectral observable (obtained with M_A: Z = M_A z)
    """
    R_dictionary = {}
    r_matrix = np.zeros((len(sigma_array), 2))
    r2_matrix = np.zeros((len(sigma_array), 2))

    R_matrix = np.zeros((len(sigma_array), 2))
    R2_matrix = np.zeros((len(sigma_array), 2))

    # 2 stands for the number of stable branches in the hysteresis
    # the first branch (j = 0) is the forward (bottom) branch
    # the second branch (j = 1) is the backward (top) branch

    n1, n2 = 1, N-1

    A = nx.to_numpy_array(nx.star_graph(n2))
    K = np.diag(np.sum(A, axis=0))
    M = np.array([np.concatenate([[1], np.zeros(n2)]),
                  np.concatenate([[0], np.ones(n2) / n2])])

    omega1 = np.diag(K)[0]
    omega2 = np.diag(K)[1]
    omega_array = np.array([omega1, omega2])
    omega = np.array([omega1] + n2 * [omega2])
    Omega = (omega1 + (N-1)*omega2)/N
    omega_CM = np.array([omega1-Omega] + n2 * [omega2-Omega])

    """ Get the bottom branch (forward branch) """
    # j = 0
    # W0 = [0.001, 0]
    # theta0 = np.linspace(0, 2*np.pi, N)
    # 
    # for i in tqdm(range(len(sigma_array))):
    #     time.sleep(1)
    # 
    #     sigma = sigma_array[i]
    # 
    #     # Integrate complete dynamics
    # 
    #     args_kuramoto = (omega_CM, N*sigma)
    #     # *N, to cancel the N in the definition of the dynamics "kuramoto"
    #     # in my code.
    #     kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
    #                                       "dop853", theta0,
    #                                       *args_kuramoto)
    # 
    #     r2 = np.absolute(
    #         np.sum(M[1, :] * np.exp(1j * kuramoto_sol),
    #                axis=1))
    #     r = np.absolute(
    #         np.sum((n1*M[0, :] + n2*M[1, :])*np.exp(1j * kuramoto_sol),
    #                axis=1)) / N
    # 
    #     r2_mean = np.sum(r2[averaging * int(t1 // dt) // 10:]
    #                      ) / len(r2[averaging * int(t1 // dt) // 10:])
    #     r_mean = np.sum(r[averaging * int(t1 // dt) // 10:]
    #                     ) / len(r[averaging * int(t1 // dt) // 10:])
    # 
    #     # Integrate reduced dynamics
    # 
    #     MAMp = multi_dot([M, A, pinv(M)])
    #     args_red_kuramoto = (N, omega1, omega2, sigma)
    #     red_kuramoto_sol = integrate_dynamics(t0_red, t1_red, dt_red,
    #                                           reduced_kuramoto_star_2D,
    #                                           MAMp, "dop853", W0,
    #                                           *args_red_kuramoto)
    # 
    #     R2 = red_kuramoto_sol[:, 0]
    #     Phi = red_kuramoto_sol[:, 1]
    # 
    #     R = np.absolute(np.exp(1j*Phi) + n2*R2) / N
    # 
    #     R2_mean = np.sum(R2[averaging * int(t1 // dt) // 10:]
    #                      )/len(R2[averaging*int(t1//dt)//10:])
    #     Phi_mean = np.sum(Phi[averaging * int(t1 // dt) // 10:]
    #                       ) / len(Phi[averaging * int(t1 // dt) // 10:])
    #     R_mean = np.sum(R[averaging*int(t1//dt)//10:]
    #                     )/len(R[averaging*int(t1//dt)//10:])
    # 
    #     r_matrix[i, j] = r_mean
    #     r2_matrix[i, j] = r2_mean
    # 
    #     R_matrix[i, j] = R_mean
    #     R2_matrix[i, j] = R2_mean
    # 
    #     W0 = [0.001, Phi_mean]
    #     theta0 = kuramoto_sol[-1, :]
    #     # np.mean(kuramoto_sol[averaging*int(t1//dt)//10:, :], axis=0)
    #     # print(np.var((theta0 - np.linspace(0, 2 * np.pi, N))%(2*np.pi)))
    #     # if i > 0:
    #     #     print(np.absolute(np.sum(np.exp(1j * theta0))) / N, r_mean)
    # 
    #     if plot_temporal_series:
    #         import matplotlib.pyplot as plt
    # 
    #         plt.figure(figsize=(8, 8))
    # 
    #         second_community_color = "#f16913"
    #         reduced_second_community_color = "#fdd0a2"
    #         ylim = [-0.02, 1.1]
    # 
    #         plt.subplot(211)
    #         plt.suptitle("$\\sigma = {}, \\omega_1 = {}, \\omega_2 = {}$"
    #                      .format(np.round(sigma_array[i], 3),
    #                              np.round(omega1, 3),
    #                              np.round(omega2, 3)), y=1.0)
    #         plt.plot(r, color="k", label="Complete spectral")
    #         plt.plot(R, color="grey", label="Reduced spectral")
    #         plt.plot(r_mean*np.ones(int(t1//dt)), color="r")
    #         plt.plot(R_mean*np.ones(int(t1//dt)), color="orange")
    #         plt.ylim(ylim)
    #         plt.ylabel("$R$", fontsize=12)
    #         plt.legend(loc=1, fontsize=10)
    # 
    #         plt.subplot(212)
    #         plt.plot(r2, color=second_community_color)
    #         plt.plot(R2, color=reduced_second_community_color)
    #         plt.ylabel("$R_2$", fontsize=12)
    #         plt.xlabel("$t$", fontsize=12)
    #         plt.ylim(ylim)
    # 
    #         plt.tight_layout()
    # 
    #         plt.show()

    """ Get the top branch (backward branch) """
    j = 1
    W0 = [0.5, 0]
    theta0 = np.random.normal(0, 0.01, N)  # np.linspace(0, 3, N)  #
    # print(np.absolute(np.mean(np.exp(1j*theta0))))

    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)

        sigma = sigma_array[-1 - i]

        # Integrate complete dynamics

        args_kuramoto = (omega_CM, N * sigma)
        # *N, to cancel the N in the definition of the dynamics "kuramoto"
        # in my code.
        kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                          "dop853", theta0,
                                          *args_kuramoto)

        r2 = np.absolute(
            np.sum(M[1, :] * np.exp(1j * kuramoto_sol),
                   axis=1))
        r = np.absolute(
            np.sum(
                (n1 * M[0, :] + n2 * M[1, :]) * np.exp(1j * kuramoto_sol),
                axis=1)) / N

        r2_mean = np.sum(r2[averaging * int(t1 // dt) // 10:]
                         ) / len(r2[averaging * int(t1 // dt) // 10:])
        r_mean = np.sum(r[averaging * int(t1 // dt) // 10:]
                        ) / len(r[averaging * int(t1 // dt) // 10:])

        # Integrate reduced dynamics

        MAMp = multi_dot([M, A, pinv(M)])
        args_red_kuramoto = (N, omega1, omega2, sigma)
        red_kuramoto_sol = integrate_dynamics(t0_red, t1_red, dt_red,
                                              reduced_kuramoto_star_2D,
                                              MAMp, "dop853", W0,
                                              *args_red_kuramoto)

        R2 = red_kuramoto_sol[:, 0]
        Phi = red_kuramoto_sol[:, 1]

        R = np.absolute(np.exp(1j * Phi) + n2 * R2) / N

        R2_mean = np.sum(R2[averaging * int(t1 // dt) // 10:]
                         ) / len(R2[averaging * int(t1 // dt) // 10:])
        Phi_mean = np.sum(Phi[averaging * int(t1 // dt) // 10:]
                          ) / len(Phi[averaging * int(t1 // dt) // 10:])
        R_mean = np.sum(R[averaging * int(t1 // dt) // 10:]
                        ) / len(R[averaging * int(t1 // dt) // 10:])

        r_matrix[-1-i, j] = r_mean
        r2_matrix[-1-i, j] = r2_mean

        R_matrix[-1-i, j] = R_mean
        R2_matrix[-1-i, j] = R2_mean

        W0 = [R_mean, Phi_mean]
        theta0 = kuramoto_sol[-1, :]

        if plot_temporal_series:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 8))

            second_community_color = "#f16913"
            reduced_second_community_color = "#fdd0a2"
            ylim = [-0.02, 1.1]

            plt.subplot(211)
            plt.suptitle("$\\sigma = {}, \\omega_1 = {}, \\omega_2 = {}$"
                         .format(np.round(sigma_array[-1 - i], 3),
                                 np.round(omega1, 3),
                                 np.round(omega2, 3)), y=1.0)
            plt.plot(r, color="k", label="Complete spectral")
            plt.plot(R, color="grey", label="Reduced spectral")
            plt.plot(r_mean * np.ones(int(t1 // dt)), color="r")
            plt.plot(R_mean * np.ones(int(t1 // dt)), color="orange")
            plt.ylim(ylim)
            plt.ylabel("$R$", fontsize=12)
            plt.legend(loc=1, fontsize=10)

            plt.subplot(212)
            plt.plot(r2, color=second_community_color)
            plt.plot(R2, color=reduced_second_community_color)
            plt.ylabel("$R_2$", fontsize=12)
            plt.xlabel("$t$", fontsize=12)
            plt.ylim(ylim)

            plt.tight_layout()

            plt.show()

    R_dictionary["r"] = r_matrix.tolist()
    R_dictionary["r2"] = r2_matrix.tolist()

    R_dictionary["R"] = R_matrix.tolist()
    R_dictionary["R2"] = R2_matrix.tolist()

    R_dictionary["sigma_array"] = sigma_array.tolist()
    R_dictionary["omega_array"] = omega_array.tolist()

    return R_dictionary


def sigma_critical(omega1, omega2, N, alpha):
    return (omega1 - omega2)/np.sqrt(N**2 - 4*(N-1)*np.sin(alpha)**2)


def get_data_kuramoto_two_triangles(sigma_array, omega1_array,
                                    t0, t1, dt,
                                    plot_temporal_series=0,
                                    plot_temporal_series_2=0):
    """
    Generate data for different reduced Kuramoto dynamics on a 2-triangle graph

    See graphs/reduction_two_triangles.py

    :param sigma_array: Array of the coupling constant sigma
    :param omega1_array: Array of the natural frequencies of the first
                         community
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :param plot_temporal_series_2: (boolean)
                                  Show temporal series (1) or not (0)
                                  of r1, R1, r2, R2, phi = phi1 - phi2,
                                  Phi = Phi1 - Phi2 when the weights are
                                  uniform in M
    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [[--- r ---]],
                               "r1",            [[--- r1---]],
                               "r2",            [[--- r2---]],
                               "r_uni",         [[--- r_uni ---]],
                               "r1_uni",        [[--- r1_uni ---]],
                               "r2_uni",        [[--- r2_uni ---]],
                               "R",             [[--- R ---]],
                               ...                ...
                               "R_uni",         [[--- R_uni ---]],
                               ...                ...
                               "sigma_array",    sigma_array,
                               "omega1_array",   omega1_array}
            where the values [[---X---]] is an array of shape
            len(sigma_list) times len(omega_1_list) of the order parameter X.
            R is the spectral observable (obtained with M_A: Z = M_A z)
            R_uni is the frequency observable (obtained with M_0 = M_W:Z=M_W z)
                                                          or M_T
    """
    R_dictionary = {}
    r_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r1_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r2_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    r_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r1_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r2_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    n1, n2 = 3, 3
    N = n1 + n2

    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)
        sigma = sigma_array[i]
        A = np.array([[0, 1, 1, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [1, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 1, 0]])
        K = np.diag(np.sum(A, axis=0))
        # P = (np.block([[np.ones(n1), np.zeros(n2)],
        #                [np.zeros(n1), np.ones(n2)]])).T
        # C = np.array([[0.4142, -0.3298],
        #               [0.4142,  0.3298]])
        # V = get_eigenvectors_matrix(A, 2)  # Not normalized
        # # Vp = pinv(V)
        # # C = np.dot(M_0, Vp)
        # # CV = np.dot(C, V)
        # CV = np.dot(C, V)
        # M = (CV.T / (np.sum(CV, axis=1))).T
        # print(M)
        # (CV.T / (np.sum(CV, axis=1))).T
        M = np.array([[0.2929, 0.2929, 0.3143, 0.0999, 0, 0],
                      [0, 0, 0.0999, 0.3143, 0.2929, 0.2929]])
        # M_0 = np.array([[1/4, 1/4, 1/4, 1/4, 0, 0],
        #                 [0, 0, 1/4, 1/4, 1/4, 1/4]])
        M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                        [np.zeros(n1), 1 / n2 * np.ones(n2)]])
        for j in range(len(omega1_array)):

            omega1 = omega1_array[j]
            omega2 = -n1/n2*omega1
            omega = np.array(n1 * [omega1] + n2 * [omega2])
            # omega_array = np.array([omega1, omega2])
            # omega = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])

            # Integrate complete dynamics

            theta0 = np.linspace(0, 2*np.pi, N)  # 2*np.pi*np.random.rand(N)
            z0 = np.exp(1j * theta0)
            args_kuramoto = (omega, sigma)
            kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                              "dop853", theta0,
                                              *args_kuramoto)

            r1 = np.absolute(
                np.sum(M[0, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r = np.absolute(
                np.sum((n1*M[0, :] + n2*M[1, :])*np.exp(1j * kuramoto_sol),
                       axis=1)) / N
            r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
            r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

            r1_uni = np.absolute(
                np.sum(M_0[0, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r2_uni = np.absolute(
                np.sum(M_0[1, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r_uni = np.absolute(
                np.sum((n1 * M_0[0, :] + n2 * M_0[1, :]) *
                       np.exp(1j * kuramoto_sol), axis=1)) / N
            r1_uni_mean = np.sum(r1_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r1_uni[5 * int(t1 // dt) // 10:])
            r2_uni_mean = np.sum(r2_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r2_uni[5 * int(t1 // dt) // 10:])
            r_uni_mean = np.sum(r_uni[5 * int(t1 // dt) // 10:]
                                ) / len(r_uni[5 * int(t1 // dt) // 10:])

            # Integrate reduced dynamics

            Z0 = np.dot(M, z0)

            # Spectral weights
            alpha = np.sum(np.dot(M, A), axis=1)
            MAMp = multi_dot([M, A, pinv(M)])
            MKMp = multi_dot([M, K, pinv(M)])  # np.diag(alpha)  #
            MWMp = multi_dot([M, np.diag(omega), pinv(M)])
            # from scipy.linalg import eig
            # print(eig(A)[0], eig(MAMp)[0],
            #       eig(multi_dot([M_0, A, pinv(M_0)]))[0])

            args_red_kuramoto_2D = (MWMp, sigma, N, MKMp, alpha)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D2,
                                                  MAMp, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1, Z2 =\
                red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1, R2 =\
                np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1*Z1 + n2*Z2)/N
            R1_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                             )/len(R1[5*int(t1//dt)//10:])
            R2_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                             )/len(R2[5*int(t1//dt)//10:])
            R_mean = np.sum(R[5*int(t1//dt)//10:]
                            )/len(R[5*int(t1//dt)//10:])

            # Uniform weights
            Z00 = np.dot(M_0, z0)
            alpha0 = np.sum(np.dot(M_0, A), axis=1)
            MAMp0 = multi_dot([M_0, A, pinv(M_0)])
            MKMp0 = multi_dot([M_0, K, pinv(M_0)])  # np.diag(alpha0) #
            MWMp0 = multi_dot([M_0, np.diag(omega), pinv(M_0)])

            args_red_kuramoto_2D = (MWMp0, sigma, N, MKMp0, alpha0)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D2,
                                                  MAMp0, "zvode", Z00,
                                                  *args_red_kuramoto_2D)
            Z1_uni, Z2_uni = \
                red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1_uni, R2_uni = \
                np.absolute(Z1_uni), np.absolute(Z2_uni)
            R_uni = np.absolute(n1 * Z1_uni + n2 * Z2_uni) / N
            R1_uni_mean = np.sum(R1_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(R1_uni[5 * int(t1 // dt) // 10:])
            R2_uni_mean = np.sum(R2_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(R2_uni[5 * int(t1 // dt) // 10:])
            R_uni_mean = np.sum(R_uni[5 * int(t1 // dt) // 10:]
                                ) / len(R_uni[5 * int(t1 // dt) // 10:])

            r_matrix[i, j] = r_mean
            r1_matrix[i, j] = r1_mean
            r2_matrix[i, j] = r2_mean

            r_uni_matrix[i, j] = r_uni_mean
            r1_uni_matrix[i, j] = r1_uni_mean
            r2_uni_matrix[i, j] = r2_uni_mean

            R_matrix[i, j] = R_mean
            R1_matrix[i, j] = R1_mean
            R2_matrix[i, j] = R2_mean

            R_uni_matrix[i, j] = R_uni_mean
            R1_uni_matrix[i, j] = R1_uni_mean
            R2_uni_matrix[i, j] = R2_uni_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 8))

                first_community_color = "#2171b5"
                reduced_first_community_color = "#9ecae1"
                second_community_color = "#f16913"
                reduced_second_community_color = "#fdd0a2"
                ylim = [-0.02, 1.1]

                plt.subplot(321)
                plt.suptitle("$\\sigma = {}, \\omega_1 = {}, \\omega_2 = {}$"
                             .format(np.round(sigma_array[i], 3),
                                     np.round(omega1_array[j], 3),
                                     np.round(-n1/n2*omega1, 3)), y=1.0)
                plt.plot(r, color="k", label="Complete spectral")
                plt.plot(R, color="grey", label="Reduced spectral")
                plt.plot(r_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R_mean*np.ones(int(t1//dt)), color="orange")
                plt.ylim(ylim)
                plt.ylabel("$R$", fontsize=12)
                plt.legend(loc=1, fontsize=10)

                plt.subplot(322)
                plt.plot(r_uni, color="k", label="Complete uniform")
                plt.plot(R_uni, color="grey", label="Reduced uniform")
                plt.plot(r_uni_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R_uni_mean*np.ones(int(t1//dt)), color="orange")
                # plt.plot(r_mean * np.ones(len(r)), color="r")
                plt.ylim(ylim)
                plt.legend(loc=1, fontsize=10)

                plt.subplot(323)
                plt.plot(r1, color=first_community_color)
                plt.plot(R1, color=reduced_first_community_color)
                plt.ylabel("$R_1$", fontsize=12)
                plt.ylim(ylim)

                plt.subplot(324)
                plt.plot(r1_uni, color=first_community_color)
                plt.plot(R1_uni, color=reduced_first_community_color)
                plt.ylim(ylim)

                plt.subplot(325)
                plt.plot(r2, color=second_community_color)
                plt.plot(R2, color=reduced_second_community_color)
                plt.ylabel("$R_2$", fontsize=12)
                plt.xlabel("$t$", fontsize=12)
                plt.ylim(ylim)

                plt.subplot(326)
                plt.plot(r2_uni, color=second_community_color)
                plt.plot(R2_uni, color=reduced_second_community_color)
                plt.ylim(ylim)
                plt.xlabel("$t$", fontsize=12)

                plt.tight_layout()

                plt.show()

            if plot_temporal_series_2:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 10))

                Phi = np.angle(Z1) - np.angle(Z2)

                phi1_uni = np.angle(
                    np.sum(M_0[0, :] * np.exp(1j * kuramoto_sol[:, :]),
                           axis=1))
                phi2_uni = np.angle(
                    np.sum(M_0[1, :] * np.exp(1j * kuramoto_sol[:, :]),
                           axis=1))
                plt.subplot(311)
                plt.plot(r1_uni, color="k")
                plt.plot(R1, color="b", linestyle='-')
                plt.ylabel("$R_1$")

                plt.subplot(312)
                plt.plot(r2_uni, color="k")
                plt.plot(R2, color="r", linestyle='-')
                plt.ylabel("$R_2$")

                plt.subplot(313)
                # plt.scatter(t0, Phi[0], color="k", s=50)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1),
                            phi1_uni - phi2_uni,
                            color="k", s=10)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1), Phi,
                            color="purple", s=5)
                plt.ylabel("$\\Phi = \\Phi_1 - \\Phi_2$")
                plt.xlabel("Time $t$")
                plt.show()

    R_dictionary["r"] = r_matrix.tolist()
    R_dictionary["r1"] = r1_matrix.tolist()
    R_dictionary["r2"] = r2_matrix.tolist()

    R_dictionary["r_uni"] = r_uni_matrix.tolist()
    R_dictionary["r1_uni"] = r1_uni_matrix.tolist()
    R_dictionary["r2_uni"] = r2_uni_matrix.tolist()

    R_dictionary["R"] = R_matrix.tolist()
    R_dictionary["R1"] = R1_matrix.tolist()
    R_dictionary["R2"] = R2_matrix.tolist()

    R_dictionary["R_uni"] = R_uni_matrix.tolist()
    R_dictionary["R1_uni"] = R1_uni_matrix.tolist()
    R_dictionary["R2_uni"] = R2_uni_matrix.tolist()

    R_dictionary["sigma_array"] = sigma_array.tolist()
    R_dictionary["omega1_array"] = omega1_array.tolist()

    return R_dictionary


def get_data_kuramoto_two_triangles_3D(sigma_array, omega1_array, averaging,
                                       t0, t1, dt,
                                       plot_temporal_series=0):
    """
    Generate data for different reduced Kuramoto dynamics on a 2-triangle graph

    See graphs/reduction_two_triangles.py

    :param sigma_array: Array of the coupling constant sigma
    :param omega1_array: Array of the natural frequencies of the first
                         community
    :param averaging: ex. averaging = 8 means that we average from 8//10 of
                          the time series to the end of the time series
                          [8//10, :]
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [[--- r ---]],
                               "r1",            [[--- r1---]],
                               "r2",            [[--- r2---]],
                               "r3",            [[--- r3---]],
                               "r_uni",         [[--- r_uni ---]],
                               "r1_uni",        [[--- r1_uni ---]],
                               "r2_uni",        [[--- r2_uni ---]],
                               "r3_uni",        [[--- r2_uni ---]],
                               "R",             [[--- R ---]],
                               ...                ...
                               "R_uni",         [[--- R_uni ---]],
                               ...                ...
                               "sigma_array",    sigma_array,
                               "omega1_array",   omega1_array}
            where the values [[---X---]] is an array of shape
            len(sigma_list) times len(omega_1_list) of the order parameter X.
            R is the spectral observable (obtained with M_A: Z = M_A z)
            R_uni is the degree observable (obtained with M_0 = M_K:Z=M_K z)
                                                          or M_T
    """
    R_dictionary = {}
    r_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r1_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r2_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r3_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    r_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r1_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r2_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r3_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R3_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R3_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    n1, n2, n3 = 2, 2, 2
    N = n1 + n2 + n3

    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)
        sigma = sigma_array[i]
        A = np.array([[0, 1, 1, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [1, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 1, 0]])
        K = np.diag(np.sum(A, axis=0))
        # P = (np.block([[np.ones(n1), np.zeros(n2), np.zeros(n3)],
        #                [np.zeros(n1), np.ones(n2), np.zeros(n3)],
        #                [np.zeros(n1), np.zeros(n2), np.zeros(n3)]])).T
        # vapvep = np.linalg.eig(A)
        # V0 = vapvep[1][:, 0]
        # V1 = vapvep[1][:, 1]
        # # V2 = vapvep[1][:, 2]
        # V3 = vapvep[1][:, 3]
        # # V4 = vapvep[1][:, 4]
        # # V5 = vapvep[1][:, 5]
        # V = np.array([V0, V1, V3])  # get_eigenvectors_matrix(A, 3)

        # One target procedure T1 = K
        # M_0 = np.array([[1/2, 1/2, 0, 0, 0, 0],
        #                 [0, 0, 1/2, 1/2, 0, 0],
        #                 [0, 0, 0, 0, 1/2, 1/2]])   # = M_T = M_K

        # One target procedure T1 = W
        M_0 = np.array([[1/3, 1/3, 1/3, 0, 0, 0],
                        [0, 0, 0, 1/2, 1/2, 0],
                        [0, 0, 0, 0, 0, 1]])   # = M_T = M_W

        # Two target  T1 = A, T2 = K
        # M = np.array([[4.47168784e-01,  4.47168784e-01,  1.44337567e-01,
        #                -1.44337567e-01, 5.28312164e-02, 5.28312164e-02],
        #               [1.50274614e-16, -1.55036718e-16,  5.00000000e-01,
        #                5.00000000e-01, -1.62588398e-17, -1.62588398e-17],
        #               [5.28312164e-02,  5.28312164e-02, -1.44337567e-01,
        #                1.44337567e-01, 4.47168784e-01, 4.47168784e-01]])

        # Three target procedure T1 = W, T2 = A, T3 = K
        # M = np.array([[3.46225045e-01, 3.46225045e-01,  3.46225045e-01,
        #                -4.57531755e-02, -4.57531755e-02,  5.28312164e-02],
        #               [1.66666667e-01, 1.66666667e-01,  1.66666667e-01,
        #                2.50000000e-01, 2.50000000e-01, -7.82845966e-17],
        #               [-1.28917115e-02, -1.28917115e-02, -1.28917115e-02,
        #                2.95753175e-01, 2.95753175e-01,  4.47168784e-01]])

        # Three target other procedure T1 = W, T2 = A, T3 = K
        # M =  np.array([[0.33018754,  0.33018754,  0.33018754,
        #                  0.01116455,  0.01116455, -0.01289171],
        #                [0.17067604,  0.17067604,  0.17067604,  0.17327057,
        #                  0.17327057, 0.14143073],
        #                [-0.00086358, -0.00086358, -0.00086358,  0.31556488,
        #                 0.31556488, 0.37146098]])

        # Three target T1 = A, T2 = K, T3 = W
        M = np.array([[0.30827989,  0.30827989,  0.27033474,
                       0.06299859,  0.02505344,  0.02505344],
                      [-0.00966878, -0.00966878,  0.15141561,
                       0.34858439,  0.25966878,  0.25966878],
                      [0.09449788,  0.09449788, -0.11383545,
                       0.11383545,  0.40550212,  0.40550212]])


        for j in range(len(omega1_array)):

            omega1 = omega1_array[j]
            omega2 = -n1/n2*omega1
            # omega3 = n1/n2*omega1
            # omega = np.array(n1 * [omega1] + n2 * [omega2] + n3 * [omega3])
            omega = np.array(3 * [omega1] + 3 * [omega2])

            # Integrate complete dynamics

            theta0 = np.linspace(0, 2*np.pi, 6)  # 2*np.pi*np.random.rand(N)
            z0 = np.exp(1j * theta0)
            args_kuramoto = (omega, sigma)
            kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                              "dop853", theta0,
                                              *args_kuramoto)

            r1 = np.absolute(
                np.sum(M[0, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r3 = np.absolute(
                np.sum(M[2, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r = np.absolute(
                np.sum((n1*M[0, :] + n2*M[1, :] + n3*M[2, :]) *
                       np.exp(1j * kuramoto_sol), axis=1)) / N
            r1_mean = np.sum(r1[averaging * int(t1 // dt) // 10:]
                             ) / len(r1[averaging * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[averaging * int(t1 // dt) // 10:]
                             ) / len(r2[averaging * int(t1 // dt) // 10:])
            r3_mean = np.sum(r3[averaging * int(t1 // dt) // 10:]
                             ) / len(r3[averaging * int(t1 // dt) // 10:])
            r_mean = np.sum(r[averaging * int(t1 // dt) // 10:]
                            ) / len(r[averaging * int(t1 // dt) // 10:])

            r1_uni = np.absolute(
                np.sum(M_0[0, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r2_uni = np.absolute(
                np.sum(M_0[1, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r3_uni = np.absolute(
                np.sum(M_0[2, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r_uni = np.absolute(
                np.sum((n1 * M_0[0, :] + n2 * M_0[1, :] + n3 * M_0[2, :]) *
                       np.exp(1j * kuramoto_sol), axis=1)) / N
            r1_uni_mean = np.sum(r1_uni[averaging * int(t1 // dt) // 10:]
                                 ) / len(r1_uni[averaging*int(t1 // dt)//10:])
            r2_uni_mean = np.sum(r2_uni[averaging * int(t1 // dt) // 10:]
                                 ) / len(r2_uni[averaging*int(t1 // dt)//10:])
            r3_uni_mean = np.sum(r3_uni[averaging * int(t1 // dt)//10:]
                                 ) / len(r3_uni[averaging*int(t1 // dt)//10:])
            r_uni_mean = np.sum(r_uni[averaging * int(t1 // dt) // 10:]
                                ) / len(r_uni[averaging * int(t1 // dt)//10:])

            # print(M, M_0, (n1*M[0, :] + n2*M[1, :] + n3*M[2, :])/N,
            #       (n1*M_0[0, :] + n2*M_0[1, :] + n3*M_0[2, :])/N)

            # Integrate reduced dynamics

            Z0 = np.dot(M, z0)

            # Spectral weights
            alpha = np.sum(np.dot(M, A), axis=1)
            MAMp = multi_dot([M, A, pinv(M)])
            MKMp = multi_dot([M, K, pinv(M)])  # np.diag(alpha) #
            MWMp = multi_dot([M, np.diag(omega), pinv(M)])

            args_red_kuramoto_2D = (MWMp, sigma, N, MKMp, alpha)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D2,
                                                  MAMp, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1, Z2, Z3 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1], \
                red_kuramoto_sol[:, 2]
            R1, R2, R3 = np.absolute(Z1), np.absolute(Z2), np.absolute(Z3)
            R = np.absolute(n1*Z1 + n2*Z2 + n3*Z3)/N
            R1_mean = np.sum(R1[averaging * int(t1 // dt) // 10:]
                             )/len(R1[averaging*int(t1//dt) // 10:])
            R2_mean = np.sum(R2[averaging * int(t1 // dt) // 10:]
                             )/len(R2[averaging*int(t1//dt) // 10:])
            R3_mean = np.sum(R3[averaging * int(t1 // dt) // 10:]
                             ) / len(R3[averaging * int(t1 // dt) // 10:])
            R_mean = np.sum(R[averaging*int(t1//dt)//10:]
                            )/len(R[averaging*int(t1//dt)//10:])

            # Uniform weights
            Z00 = np.dot(M_0, z0)
            alpha0 = np.sum(np.dot(M_0, A), axis=1)
            MAMp0 = multi_dot([M_0, A, pinv(M_0)])
            MKMp0 = multi_dot([M_0, K, pinv(M_0)])  # np.diag(alpha0) #
            MWMp0 = multi_dot([M_0, np.diag(omega), pinv(M_0)])

            args_red_kuramoto_2D = (MWMp0, sigma, N, MKMp0, alpha0)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D2,
                                                  MAMp0, "zvode", Z00,
                                                  *args_red_kuramoto_2D)
            Z1_uni, Z2_uni, Z3_uni = red_kuramoto_sol[:, 0], \
                red_kuramoto_sol[:, 1], red_kuramoto_sol[:, 2]
            R1_uni, R2_uni, R3_uni = \
                np.absolute(Z1_uni), np.absolute(Z2_uni), np.absolute(Z3_uni)
            R_uni = np.absolute(n1*Z1_uni + n2*Z2_uni + n3*Z3_uni) / N
            R1_uni_mean = np.sum(R1_uni[averaging * int(t1 // dt)//10:]
                                 ) / len(R1_uni[averaging*int(t1 // dt)//10:])
            R2_uni_mean = np.sum(R2_uni[averaging * int(t1 // dt)//10:]
                                 ) / len(R2_uni[averaging*int(t1 // dt)//10:])
            R3_uni_mean = np.sum(R3_uni[averaging * int(t1 // dt)//10:]
                                 ) / len(R3_uni[averaging*int(t1 // dt)//10:])
            R_uni_mean = np.sum(R_uni[averaging * int(t1 // dt) // 10:]
                                ) / len(R_uni[averaging * int(t1 // dt)//10:])

            r_matrix[i, j] = r_mean
            r1_matrix[i, j] = r1_mean
            r2_matrix[i, j] = r2_mean
            r3_matrix[i, j] = r3_mean

            r_uni_matrix[i, j] = r_uni_mean
            r1_uni_matrix[i, j] = r1_uni_mean
            r2_uni_matrix[i, j] = r2_uni_mean
            r3_uni_matrix[i, j] = r3_uni_mean

            R_matrix[i, j] = R_mean
            R1_matrix[i, j] = R1_mean
            R2_matrix[i, j] = R2_mean
            R3_matrix[i, j] = R3_mean

            R_uni_matrix[i, j] = R_uni_mean
            R1_uni_matrix[i, j] = R1_uni_mean
            R2_uni_matrix[i, j] = R2_uni_mean
            R3_uni_matrix[i, j] = R3_uni_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 8))

                first_community_color = "#2171b5"
                reduced_first_community_color = "#9ecae1"
                second_community_color = "#f16913"
                reduced_second_community_color = "#fdd0a2"
                third_community_color = "#4a1486"
                reduced_third_community_color = "#9e9ac8"

                ylim = [-0.02, 1.1]

                plt.subplot(421)
                # plt.suptitle("$\\sigma = {}, \\omega_1 = {}, \\omega_2 = {}$,
                #  "
                #              "$\\omega_3 = {}$"
                #              .format(np.round(sigma_array[i], 3),
                #                      np.round(omega1, 3),
                #                      np.round(omega2, 3),
                #                      np.round(omega3, 3), y=1.0))
                plt.plot(r, color="k", label="Complete spectral")
                plt.plot(R, color="grey", label="Reduced spectral")
                plt.plot(r_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R_mean*np.ones(int(t1//dt)), color="orange")
                plt.ylim(ylim)
                plt.ylabel("$R$", fontsize=12)
                plt.legend(loc=4, fontsize=10)

                plt.subplot(422)
                plt.plot(r_uni, color="k", label="Complete uniform")
                plt.plot(R_uni, color="grey", label="Reduced uniform")
                plt.plot(r_uni_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R_uni_mean*np.ones(int(t1//dt)), color="orange")
                # plt.plot(r_mean * np.ones(len(r)), color="r")
                plt.ylim(ylim)
                plt.legend(loc=4, fontsize=10)

                plt.subplot(423)
                plt.plot(r1, color=first_community_color)
                plt.plot(R1, color=reduced_first_community_color)
                plt.plot(r1_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R1_mean*np.ones(int(t1//dt)), color="orange")
                plt.ylabel("$R_1$", fontsize=12)
                plt.ylim(ylim)

                plt.subplot(424)
                plt.plot(r1_uni, color=first_community_color)
                plt.plot(R1_uni, color=reduced_first_community_color)
                plt.plot(r1_uni_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R1_uni_mean*np.ones(int(t1//dt)), color="orange")
                plt.ylim(ylim)

                plt.subplot(425)
                plt.plot(r2, color=second_community_color)
                plt.plot(R2, color=reduced_second_community_color)
                plt.plot(r2_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R2_mean*np.ones(int(t1//dt)), color="orange")
                plt.ylabel("$R_2$", fontsize=12)
                plt.ylim(ylim)

                plt.subplot(426)
                plt.plot(r2_uni, color=second_community_color)
                plt.plot(R2_uni, color=reduced_second_community_color)
                plt.plot(r2_uni_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R2_uni_mean*np.ones(int(t1//dt)), color="orange")
                plt.ylim(ylim)

                plt.subplot(427)
                plt.plot(r3, color=third_community_color)
                plt.plot(R3, color=reduced_third_community_color)
                plt.plot(r3_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R3_mean*np.ones(int(t1//dt)), color="orange")
                plt.ylabel("$R_3$", fontsize=12)
                plt.xlabel("$t$", fontsize=12)
                plt.ylim(ylim)

                plt.subplot(428)
                plt.plot(r3_uni, color=third_community_color)
                plt.plot(R3_uni, color=reduced_third_community_color)
                plt.plot(r3_uni_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R3_uni_mean*np.ones(int(t1//dt)), color="orange")
                plt.ylim(ylim)
                plt.xlabel("$t$", fontsize=12)

                plt.tight_layout()

                plt.show()

    R_dictionary["r"] = r_matrix.tolist()
    R_dictionary["r1"] = r1_matrix.tolist()
    R_dictionary["r2"] = r2_matrix.tolist()
    R_dictionary["r3"] = r3_matrix.tolist()

    R_dictionary["r_uni"] = r_uni_matrix.tolist()
    R_dictionary["r1_uni"] = r1_uni_matrix.tolist()
    R_dictionary["r2_uni"] = r2_uni_matrix.tolist()
    R_dictionary["r3_uni"] = r3_uni_matrix.tolist()

    R_dictionary["R"] = R_matrix.tolist()
    R_dictionary["R1"] = R1_matrix.tolist()
    R_dictionary["R2"] = R2_matrix.tolist()
    R_dictionary["R3"] = R3_matrix.tolist()

    R_dictionary["R_uni"] = R_uni_matrix.tolist()
    R_dictionary["R1_uni"] = R1_uni_matrix.tolist()
    R_dictionary["R2_uni"] = R2_uni_matrix.tolist()
    R_dictionary["R3_uni"] = R3_uni_matrix.tolist()

    R_dictionary["sigma_array"] = sigma_array.tolist()
    R_dictionary["omega1_array"] = omega1_array.tolist()

    return R_dictionary


def get_data_kuramoto_small_bipartite(sigma_array, omega1_array,
                                      t0, t1, dt,
                                      plot_temporal_series=0,
                                      plot_temporal_series_2=0):
    """
    Generate data for different reduced Kuramoto dynamics on a small SBM

    See graphs/reduction_small_bipartite.py

    :param sigma_array: Array of the coupling constant sigma
    :param omega1_array: Array of the natural frequencies of the first
                         community
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :param plot_temporal_series_2: (boolean)
                                  Show temporal series (1) or not (0)
                                  of r1, R1, r2, R2, phi = phi1 - phi2,
                                  Phi = Phi1 - Phi2 when the weights are
                                  uniform in M
    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [[--- r ---]],
                               "r1",            [[--- r1---]],
                               "r2",            [[--- r2---]],
                               "r_uni",         [[--- r_uni ---]],
                               "r1_uni",        [[--- r1_uni ---]],
                               "r2_uni",        [[--- r2_uni ---]],
                               "R",             [[--- R ---]],
                               ...                ...
                               "R_uni",         [[--- R_uni ---]],
                               ...                ...
                               "sigma_array",    sigma_array,
                               "omega1_array",   omega1_array}
            where the values [[---X---]] is an array of shape
            len(sigma_list) times len(omega_1_list) of the order parameter X.
            R is the spectral observable (obtained with M_A: Z = M_A z)
            R_uni is the frequency observable (obtained with M_0 = M_W:Z=M_W z)
                                                          or M_T
    """
    R_dictionary = {}
    r_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r1_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r2_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    r_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r1_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r2_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    n1, n2 = 3, 3
    N = n1 + n2

    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)
        sigma = sigma_array[i]
        A = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1],
                      [1, 1, 0, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0]])
        K = np.diag(np.sum(A, axis=0))
        # P = (np.block([[np.ones(n1), np.zeros(n2)],
        #                [np.zeros(n1), np.ones(n2)]])).T
        # C = np.array([[0.4142, -0.3298],
        #               [0.4142,  0.3298]])
        # V = get_eigenvectors_matrix(A, 2)  # Not normalized
        # # Vp = pinv(V)
        # # C = np.dot(M_0, Vp)
        # # CV = np.dot(C, V)
        # CV = np.dot(C, V)
        # M = (CV.T / (np.sum(CV, axis=1))).T
        # print(M)
        # (CV.T / (np.sum(CV, axis=1))).T

        # See graphs/reduction_small_bipartite.py

        M = np.array([[1.33974596e-01, 0.5, 3.66025404e-01, 0.0, 0.0, 0.0],
                      [0, 0, 0, 2.67949192e-01,
                       3.66025404e-01, 3.66025404e-01]])
        M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                        [np.zeros(n1), 1 / n2 * np.ones(n2)]])
        # M_0 = M_W, the weights for a frequency reduction

        for j in range(len(omega1_array)):

            omega1 = omega1_array[j]
            omega2 = -n1/n2*omega1
            omega = np.array(n1 * [omega1] + n2 * [omega2])
            # omega_array = np.array([omega1, omega2])
            # omega = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])

            # Integrate complete dynamics

            theta0 = np.linspace(0, 2*np.pi, N)  # 2*np.pi*np.random.rand(N)
            z0 = np.exp(1j * theta0)
            args_kuramoto = (omega, sigma)
            kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                              "vode", theta0,
                                              *args_kuramoto)

            r1 = np.absolute(
                np.sum(M[0, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r = np.absolute(
                np.sum((n1*M[0, :] + n2*M[1, :])*np.exp(1j * kuramoto_sol),
                       axis=1)) / N
            r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
            r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

            r1_uni = np.absolute(
                np.sum(M_0[0, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r2_uni = np.absolute(
                np.sum(M_0[1, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r_uni = np.absolute(
                np.sum((n1 * M_0[0, :] + n2 * M_0[1, :]) *
                       np.exp(1j * kuramoto_sol), axis=1)) / N
            r1_uni_mean = np.sum(r1_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r1_uni[5 * int(t1 // dt) // 10:])
            r2_uni_mean = np.sum(r2_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r2_uni[5 * int(t1 // dt) // 10:])
            r_uni_mean = np.sum(r_uni[5 * int(t1 // dt) // 10:]
                                ) / len(r_uni[5 * int(t1 // dt) // 10:])

            # Integrate reduced dynamics

            Z0 = np.dot(M, z0)

            # Spectral weights
            alpha = np.sum(np.dot(M, A), axis=1)
            MAMp = multi_dot([M, A, pinv(M)])
            MKMp = multi_dot([M, K, pinv(M)])  # np.diag(alpha)  #
            MWMp = multi_dot([M, np.diag(omega), pinv(M)])
            # from scipy.linalg import eig
            # print(eig(A)[0], eig(MAMp)[0],
            #       eig(multi_dot([M_0, A, pinv(M_0)]))[0])

            args_red_kuramoto_2D = (MWMp, sigma, N, MKMp, alpha)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D2,
                                                  MAMp, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1, Z2 =\
                red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1, R2 =\
                np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1*Z1 + n2*Z2)/N
            R1_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                             )/len(R1[5*int(t1//dt)//10:])
            R2_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                             )/len(R2[5*int(t1//dt)//10:])
            R_mean = np.sum(R[5*int(t1//dt)//10:]
                            )/len(R[5*int(t1//dt)//10:])

            # Uniform weights (weights for the frequency reduction)
            Z00 = np.dot(M_0, z0)
            alpha0 = np.sum(np.dot(M_0, A), axis=1)
            MAMp0 = multi_dot([M_0, A, pinv(M_0)])
            MKMp0 = multi_dot([M_0, K, pinv(M_0)])  # np.diag(alpha0) #
            MWMp0 = multi_dot([M_0, np.diag(omega), pinv(M_0)])

            args_red_kuramoto_2D = (MWMp0, sigma, N, MKMp0, alpha0)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D2,
                                                  MAMp0, "zvode", Z00,
                                                  *args_red_kuramoto_2D)
            Z1_uni, Z2_uni = \
                red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1_uni, R2_uni = \
                np.absolute(Z1_uni), np.absolute(Z2_uni)
            R_uni = np.absolute(n1 * Z1_uni + n2 * Z2_uni) / N
            R1_uni_mean = np.sum(R1_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(R1_uni[5 * int(t1 // dt) // 10:])
            R2_uni_mean = np.sum(R2_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(R2_uni[5 * int(t1 // dt) // 10:])
            R_uni_mean = np.sum(R_uni[5 * int(t1 // dt) // 10:]
                                ) / len(R_uni[5 * int(t1 // dt) // 10:])

            r_matrix[i, j] = r_mean
            r1_matrix[i, j] = r1_mean
            r2_matrix[i, j] = r2_mean

            r_uni_matrix[i, j] = r_uni_mean
            r1_uni_matrix[i, j] = r1_uni_mean
            r2_uni_matrix[i, j] = r2_uni_mean

            R_matrix[i, j] = R_mean
            R1_matrix[i, j] = R1_mean
            R2_matrix[i, j] = R2_mean

            R_uni_matrix[i, j] = R_uni_mean
            R1_uni_matrix[i, j] = R1_uni_mean
            R2_uni_matrix[i, j] = R2_uni_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 8))

                first_community_color = "#2171b5"
                reduced_first_community_color = "#9ecae1"
                second_community_color = "#f16913"
                reduced_second_community_color = "#fdd0a2"
                ylim = [-0.02, 1.1]

                plt.subplot(321)
                plt.suptitle("$\\sigma = {}, \\omega_1 = {}, \\omega_2 = {}$"
                             .format(np.round(sigma_array[i], 3),
                                     np.round(omega1_array[j], 3),
                                     np.round(-n1/n2*omega1, 3)), y=1.0)
                plt.plot(r, color="k", label="Complete spectral")
                plt.plot(R, color="grey", label="Reduced spectral")
                plt.plot(r_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R_mean*np.ones(int(t1//dt)), color="orange")
                plt.ylim(ylim)
                plt.ylabel("$R$", fontsize=12)
                plt.legend(loc=1, fontsize=10)

                plt.subplot(322)
                plt.plot(r_uni, color="k", label="Complete uniform")
                plt.plot(R_uni, color="grey", label="Reduced uniform")
                plt.plot(r_uni_mean*np.ones(int(t1//dt)), color="r")
                plt.plot(R_uni_mean*np.ones(int(t1//dt)), color="orange")
                # plt.plot(r_mean * np.ones(len(r)), color="r")
                plt.ylim(ylim)
                plt.legend(loc=1, fontsize=10)

                plt.subplot(323)
                plt.plot(r1, color=first_community_color)
                plt.plot(R1, color=reduced_first_community_color)
                plt.ylabel("$R_1$", fontsize=12)
                plt.ylim(ylim)

                plt.subplot(324)
                plt.plot(r1_uni, color=first_community_color)
                plt.plot(R1_uni, color=reduced_first_community_color)
                plt.ylim(ylim)

                plt.subplot(325)
                plt.plot(r2, color=second_community_color)
                plt.plot(R2, color=reduced_second_community_color)
                plt.ylabel("$R_2$", fontsize=12)
                plt.xlabel("$t$", fontsize=12)
                plt.ylim(ylim)

                plt.subplot(326)
                plt.plot(r2_uni, color=second_community_color)
                plt.plot(R2_uni, color=reduced_second_community_color)
                plt.ylim(ylim)
                plt.xlabel("$t$", fontsize=12)

                plt.tight_layout()

                plt.show()

            if plot_temporal_series_2:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 10))

                Phi = np.angle(Z1) - np.angle(Z2)

                phi1_uni = np.angle(
                    np.sum(M_0[0, :] * np.exp(1j * kuramoto_sol[:, :]),
                           axis=1))
                phi2_uni = np.angle(
                    np.sum(M_0[1, :] * np.exp(1j * kuramoto_sol[:, :]),
                           axis=1))
                plt.subplot(311)
                plt.plot(r1_uni, color="k")
                plt.plot(R1, color="b", linestyle='-')
                plt.ylabel("$R_1$")

                plt.subplot(312)
                plt.plot(r2_uni, color="k")
                plt.plot(R2, color="r", linestyle='-')
                plt.ylabel("$R_2$")

                plt.subplot(313)
                # plt.scatter(t0, Phi[0], color="k", s=50)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1),
                            phi1_uni - phi2_uni,
                            color="k", s=10)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1), Phi,
                            color="purple", s=5)
                plt.ylabel("$\\Phi = \\Phi_1 - \\Phi_2$")
                plt.xlabel("Time $t$")
                plt.show()

    R_dictionary["r"] = r_matrix.tolist()
    R_dictionary["r1"] = r1_matrix.tolist()
    R_dictionary["r2"] = r2_matrix.tolist()

    R_dictionary["r_uni"] = r_uni_matrix.tolist()
    R_dictionary["r1_uni"] = r1_uni_matrix.tolist()
    R_dictionary["r2_uni"] = r2_uni_matrix.tolist()

    R_dictionary["R"] = R_matrix.tolist()
    R_dictionary["R1"] = R1_matrix.tolist()
    R_dictionary["R2"] = R2_matrix.tolist()

    R_dictionary["R_uni"] = R_uni_matrix.tolist()
    R_dictionary["R1_uni"] = R1_uni_matrix.tolist()
    R_dictionary["R2_uni"] = R2_uni_matrix.tolist()

    R_dictionary["sigma_array"] = sigma_array.tolist()
    R_dictionary["omega1_array"] = omega1_array.tolist()

    return R_dictionary


def get_data_kuramoto_small_bipartite_3D(sigma_array, omega1_array, averaging,
                                         t0, t1, dt,
                                         plot_temporal_series=0):
    """
    Generate data for different reduced Kuramoto dynamics on a small SBM

    See graphs/reduction_small_bipartite.py

    :param sigma_array: Array of the coupling constant sigma
    :param omega1_array: Array of the natural frequencies of the first
                         community
    :param averaging: ex. averaging = 8 means that we average from 8//10 of
                          the time series to the end of the time series
                          [8//10, :]
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [[--- r ---]],
                               "r1",            [[--- r1---]],
                               "r2",            [[--- r2---]],
                               "r3",            [[--- r3---]],
                               "r_uni",         [[--- r_uni ---]],
                               "r1_uni",        [[--- r1_uni ---]],
                               "r2_uni",        [[--- r2_uni ---]],
                               "r3_uni",        [[--- r2_uni ---]],
                               "R",             [[--- R ---]],
                               ...                ...
                               "R_uni",         [[--- R_uni ---]],
                               ...                ...
                               "sigma_array",    sigma_array,
                               "omega1_array",   omega1_array}
            where the values [[---X---]] is an array of shape
            len(sigma_list) times len(omega_1_list) of the order parameter X.
            R is the spectral observable (obtained with M_A: Z = M_A z)
            R_uni is the degree observable (obtained with M_0 = M_K:Z=M_K z)
                                                          or M_T
    """
    R_dictionary = {}
    r_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r1_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r2_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r3_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    r_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r1_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r2_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r3_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R3_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R3_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    n1, n2, n3 = 1, 1, 4
    N = n1 + n2 + n3

    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)
        sigma = sigma_array[i]
        A = np.array([[0, 0, 0, 1, 0, 0], 
                      [0, 0, 0, 1, 1, 1], 
                      [0, 0, 0, 0, 1, 1], 
                      [1, 1, 0, 0, 0, 0], 
                      [0, 1, 1, 0, 0, 0], 
                      [0, 1, 1, 0, 0, 0]])
        K = np.diag(np.sum(A, axis=0))
        M_0 = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1/4, 1/4, 1/4, 1/4]])  # = M_T = M_K

        # See graphs/reduction_small_bipartite.py

        # When we choose the three dominant eigenvectors
        # i.e. V0, V1, V2 in graphs/reduction_small_bipartite.py
        M = np.array([[0.59892362,  0.42099162, -0.177932,
                       0.58972654, -0.2158549, -0.2158549],
                      [0.19028505,  0.49043122,  0.30014617,
                       0.07142238, -0.02614241, -0.02614241],
                      [-0.00316048,  0.112102,  0.11526249,
                       0.17049174,  0.30265212,  0.30265212]])

        # When we choose the positive first dominant, the second dominant and 0
        # i.e. V0, V1, V5 in graphs/reduction_small_bipartite.py
        # M = np.array([[0.70787939, -0.17696985,  0.17696985,
        #                0.42342583, -0.06565261, -0.06565261],
        #               [-0.16102861,  0.64411444, -0.16102861,
        #                0.26580673,  0.20606803,  0.20606803],
        #               [0.1105392,  0.13386726,  0.27354127,
        #                0.09416508,  0.1939436,  0.1939436]])
        for j in range(len(omega1_array)):

            omega1 = omega1_array[j]
            omega2 = -n1 / n2 * omega1
            # omega3 = n1/n2*omega1
            # omega = np.array(n1 * [omega1] + n2 * [omega2] + n3 * [omega3])
            omega =  np.array(3 * [omega1] + 3 * [omega2])  # opposite freq.
            # omega = np.diag(K)/10  # degree-freq correlation case

            # Integrate complete dynamics

            theta0 = np.linspace(0, 2*np.pi, 6)  # 2*np.pi*np.random.rand(N)
            z0 = np.exp(1j * theta0)
            args_kuramoto = (omega, sigma)
            kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                              "dop853", theta0,
                                              *args_kuramoto)

            r1 = np.absolute(
                np.sum(M[0, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r3 = np.absolute(
                np.sum(M[2, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r = np.absolute(
                np.sum((n1 * M[0, :] + n2 * M[1, :] + n3 * M[2, :]) *
                       np.exp(1j * kuramoto_sol), axis=1)) / N
            r1_mean = np.sum(r1[averaging * int(t1 // dt) // 10:]
                             ) / len(r1[averaging * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[averaging * int(t1 // dt) // 10:]
                             ) / len(r2[averaging * int(t1 // dt) // 10:])
            r3_mean = np.sum(r3[averaging * int(t1 // dt) // 10:]
                             ) / len(r3[averaging * int(t1 // dt) // 10:])
            r_mean = np.sum(r[averaging * int(t1 // dt) // 10:]
                            ) / len(r[averaging * int(t1 // dt) // 10:])

            r1_uni = np.absolute(
                np.sum(M_0[0, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r2_uni = np.absolute(
                np.sum(M_0[1, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r3_uni = np.absolute(
                np.sum(M_0[2, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r_uni = np.absolute(
                np.sum((n1 * M_0[0, :] + n2 * M_0[1, :] + n3 * M_0[2, :]) *
                       np.exp(1j * kuramoto_sol), axis=1)) / N
            r1_uni_mean = np.sum(r1_uni[averaging * int(t1 // dt) // 10:]
                                 ) / len(
                r1_uni[averaging * int(t1 // dt) // 10:])
            r2_uni_mean = np.sum(r2_uni[averaging * int(t1 // dt) // 10:]
                                 ) / len(
                r2_uni[averaging * int(t1 // dt) // 10:])
            r3_uni_mean = np.sum(r3_uni[averaging * int(t1 // dt) // 10:]
                                 ) / len(
                r3_uni[averaging * int(t1 // dt) // 10:])
            r_uni_mean = np.sum(r_uni[averaging * int(t1 // dt) // 10:]
                                ) / len(
                r_uni[averaging * int(t1 // dt) // 10:])

            # print(M, M_0, (n1*M[0, :] + n2*M[1, :] + n3*M[2, :])/N,
            #       (n1*M_0[0, :] + n2*M_0[1, :] + n3*M_0[2, :])/N)

            # Integrate reduced dynamics

            Z0 = np.dot(M, z0)

            # Spectral weights
            alpha = np.sum(np.dot(M, A), axis=1)
            MAMp = multi_dot([M, A, pinv(M)])
            MKMp = multi_dot(
                [M, K, pinv(M)])  # np.diag(alpha) #
            MWMp = multi_dot([M, np.diag(omega), pinv(M)])

            args_red_kuramoto_2D = (MWMp, sigma, N, MKMp, alpha)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D2,
                                                  MAMp, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1, Z2, Z3 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1], \
                red_kuramoto_sol[:, 2]
            R1, R2, R3 = np.absolute(Z1), np.absolute(Z2), np.absolute(Z3)
            R = np.absolute(n1 * Z1 + n2 * Z2 + n3 * Z3) / N
            R1_mean = np.sum(R1[averaging * int(t1 // dt) // 10:]
                             ) / len(R1[averaging * int(t1 // dt) // 10:])
            R2_mean = np.sum(R2[averaging * int(t1 // dt) // 10:]
                             ) / len(R2[averaging * int(t1 // dt) // 10:])
            R3_mean = np.sum(R3[averaging * int(t1 // dt) // 10:]
                             ) / len(R3[averaging * int(t1 // dt) // 10:])
            R_mean = np.sum(R[averaging * int(t1 // dt) // 10:]
                            ) / len(R[averaging * int(t1 // dt) // 10:])

            # Uniform weights
            Z00 = np.dot(M_0, z0)
            alpha0 = np.sum(np.dot(M_0, A), axis=1)
            MAMp0 = multi_dot([M_0, A, pinv(M_0)])
            MKMp0 = multi_dot(
                [M_0, K, pinv(M_0)])  # np.diag(alpha0) #
            MWMp0 = multi_dot([M_0, np.diag(omega), pinv(M_0)])

            args_red_kuramoto_2D = (MWMp0, sigma, N, MKMp0, alpha0)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D2,
                                                  MAMp0, "zvode", Z00,
                                                  *args_red_kuramoto_2D)
            Z1_uni, Z2_uni, Z3_uni = red_kuramoto_sol[:, 0], \
                red_kuramoto_sol[:, 1], red_kuramoto_sol[:, 2]
            R1_uni, R2_uni, R3_uni = \
                np.absolute(Z1_uni), np.absolute(Z2_uni), np.absolute(Z3_uni)
            R_uni = np.absolute(n1 * Z1_uni + n2 * Z2_uni + n3 * Z3_uni) / N
            R1_uni_mean = np.sum(R1_uni[averaging * int(t1 // dt) // 10:]
                                 ) / len(
                R1_uni[averaging * int(t1 // dt) // 10:])
            R2_uni_mean = np.sum(R2_uni[averaging * int(t1 // dt) // 10:]
                                 ) / len(
                R2_uni[averaging * int(t1 // dt) // 10:])
            R3_uni_mean = np.sum(R3_uni[averaging * int(t1 // dt) // 10:]
                                 ) / len(
                R3_uni[averaging * int(t1 // dt) // 10:])
            R_uni_mean = np.sum(R_uni[averaging * int(t1 // dt) // 10:]
                                ) / len(
                R_uni[averaging * int(t1 // dt) // 10:])

            r_matrix[i, j] = r_mean
            r1_matrix[i, j] = r1_mean
            r2_matrix[i, j] = r2_mean
            r3_matrix[i, j] = r3_mean

            r_uni_matrix[i, j] = r_uni_mean
            r1_uni_matrix[i, j] = r1_uni_mean
            r2_uni_matrix[i, j] = r2_uni_mean
            r3_uni_matrix[i, j] = r3_uni_mean

            R_matrix[i, j] = R_mean
            R1_matrix[i, j] = R1_mean
            R2_matrix[i, j] = R2_mean
            R3_matrix[i, j] = R3_mean

            R_uni_matrix[i, j] = R_uni_mean
            R1_uni_matrix[i, j] = R1_uni_mean
            R2_uni_matrix[i, j] = R2_uni_mean
            R3_uni_matrix[i, j] = R3_uni_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 8))

                first_community_color = "#2171b5"
                reduced_first_community_color = "#9ecae1"
                second_community_color = "#f16913"
                reduced_second_community_color = "#fdd0a2"
                third_community_color = "#4a1486"
                reduced_third_community_color = "#9e9ac8"

                ylim = [-0.02, 1.1]

                plt.subplot(421)
                # plt.suptitle("$\\sigma = {}, \\omega_1 = {}, \\omega_2 = {}$,
                #  "
                #              "$\\omega_3 = {}$"
                #              .format(np.round(sigma_array[i], 3),
                #                      np.round(omega1, 3),
                #                      np.round(omega2, 3),
                #                      np.round(omega3, 3), y=1.0))
                plt.plot(r, color="k", label="Complete spectral")
                plt.plot(R, color="grey", label="Reduced spectral")
                plt.plot(r_mean * np.ones(int(t1 // dt)), color="r")
                plt.plot(R_mean * np.ones(int(t1 // dt)), color="orange")
                plt.ylim(ylim)
                plt.ylabel("$R$", fontsize=12)
                plt.legend(loc=4, fontsize=10)

                plt.subplot(422)
                plt.plot(r_uni, color="k", label="Complete uniform")
                plt.plot(R_uni, color="grey", label="Reduced uniform")
                plt.plot(r_uni_mean * np.ones(int(t1 // dt)), color="r")
                plt.plot(R_uni_mean * np.ones(int(t1 // dt)), color="orange")
                # plt.plot(r_mean * np.ones(len(r)), color="r")
                plt.ylim(ylim)
                plt.legend(loc=4, fontsize=10)

                plt.subplot(423)
                plt.plot(r1, color=first_community_color)
                plt.plot(R1, color=reduced_first_community_color)
                plt.plot(r1_mean * np.ones(int(t1 // dt)), color="r")
                plt.plot(R1_mean * np.ones(int(t1 // dt)), color="orange")
                plt.ylabel("$R_1$", fontsize=12)
                plt.ylim(ylim)

                plt.subplot(424)
                plt.plot(r1_uni, color=first_community_color)
                plt.plot(R1_uni, color=reduced_first_community_color)
                plt.plot(r1_uni_mean * np.ones(int(t1 // dt)), color="r")
                plt.plot(R1_uni_mean * np.ones(int(t1 // dt)), color="orange")
                plt.ylim(ylim)

                plt.subplot(425)
                plt.plot(r2, color=second_community_color)
                plt.plot(R2, color=reduced_second_community_color)
                plt.plot(r2_mean * np.ones(int(t1 // dt)), color="r")
                plt.plot(R2_mean * np.ones(int(t1 // dt)), color="orange")
                plt.ylabel("$R_2$", fontsize=12)
                plt.ylim(ylim)

                plt.subplot(426)
                plt.plot(r2_uni, color=second_community_color)
                plt.plot(R2_uni, color=reduced_second_community_color)
                plt.plot(r2_uni_mean * np.ones(int(t1 // dt)), color="r")
                plt.plot(R2_uni_mean * np.ones(int(t1 // dt)), color="orange")
                plt.ylim(ylim)

                plt.subplot(427)
                plt.plot(r3, color=third_community_color)
                plt.plot(R3, color=reduced_third_community_color)
                plt.plot(r3_mean * np.ones(int(t1 // dt)), color="r")
                plt.plot(R3_mean * np.ones(int(t1 // dt)), color="orange")
                plt.ylabel("$R_3$", fontsize=12)
                plt.xlabel("$t$", fontsize=12)
                plt.ylim(ylim)

                plt.subplot(428)
                plt.plot(r3_uni, color=third_community_color)
                plt.plot(R3_uni, color=reduced_third_community_color)
                plt.plot(r3_uni_mean * np.ones(int(t1 // dt)), color="r")
                plt.plot(R3_uni_mean * np.ones(int(t1 // dt)), color="orange")
                plt.ylim(ylim)
                plt.xlabel("$t$", fontsize=12)

                plt.tight_layout()

                plt.show()

    R_dictionary["r"] = r_matrix.tolist()
    R_dictionary["r1"] = r1_matrix.tolist()
    R_dictionary["r2"] = r2_matrix.tolist()
    R_dictionary["r3"] = r3_matrix.tolist()

    R_dictionary["r_uni"] = r_uni_matrix.tolist()
    R_dictionary["r1_uni"] = r1_uni_matrix.tolist()
    R_dictionary["r2_uni"] = r2_uni_matrix.tolist()
    R_dictionary["r3_uni"] = r3_uni_matrix.tolist()

    R_dictionary["R"] = R_matrix.tolist()
    R_dictionary["R1"] = R1_matrix.tolist()
    R_dictionary["R2"] = R2_matrix.tolist()
    R_dictionary["R3"] = R3_matrix.tolist()

    R_dictionary["R_uni"] = R_uni_matrix.tolist()
    R_dictionary["R1_uni"] = R1_uni_matrix.tolist()
    R_dictionary["R2_uni"] = R2_uni_matrix.tolist()
    R_dictionary["R3_uni"] = R3_uni_matrix.tolist()

    R_dictionary["sigma_array"] = sigma_array.tolist()
    R_dictionary["omega1_array"] = omega1_array.tolist()

    return R_dictionary


def get_data_kuramoto_bipartite_vs_N(p_out_array, omega1_array, N_array,
                                     t0, t1, dt,
                                     plot_temporal_series=0):
    """
    Generate data for different reduced Kuramoto dynamics on a random SBM

    :param p_out_array: Array of the coupling constant sigma
    :param omega1_array: Array of the natural frequencies of the first
                         community
    :param N_array: Array of the size on the graph
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r{}".format(N),             [[--- r ---]],
                               "r1{}".format(N),            [[--- r1---]],
                               "r2{}".format(N),            [[--- r2---]],
                               #"r_uni{}".format(N),       [[--- r_uni ---]],
                               #"r1_uni{}".format(N),      [[--- r1_uni ---]],
                               #"r2_uni{}".format(N),      [[--- r2_uni ---]],
                               "R{}".format(N),             [[--- R ---]],
                               ...                ...
                               #"R_uni{}".format(N),         [[--- R_uni ---]],
                               ...                ...
                               "p_out_array",    p_out_array,
                               "omega1_array",   omega1_array
                               "N_array",   N_array}
            where the values [[---X---]] is an array of shape
            len(sigma_list) times len(omega_1_list) of the order parameter X.
            R is the spectral observable (obtained with M_A: Z = M_A z)
            R_uni is the frequency observable (obtained with M_0 = M_W:Z=M_W z)
                                                          or M_T
    """
    R_dictionary = {}

    sigma = 1

    for N in N_array:

        n1 = N//2
        n2 = N - n1
        sizes = [n1, n2]

        r_matrix = np.zeros((len(p_out_array), len(omega1_array)))
        r1_matrix = np.zeros((len(p_out_array), len(omega1_array)))
        r2_matrix = np.zeros((len(p_out_array), len(omega1_array)))

        # r_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
        # r1_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
        # r2_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))

        R_matrix = np.zeros((len(p_out_array), len(omega1_array)))
        R1_matrix = np.zeros((len(p_out_array), len(omega1_array)))
        R2_matrix = np.zeros((len(p_out_array), len(omega1_array)))

        # R_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
        # R1_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
        # R2_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))

        for i in tqdm(range(len(p_out_array))):

            time.sleep(1)

            p_out = p_out_array[i]
            pq = [[0, p_out], [p_out, 0]]

            M_0 = np.block([[1/n1*np.ones(n1), np.zeros(n2)],
                            [np.zeros(n1), 1/n2*np.ones(n2)]])
            A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
            K = np.diag(np.sum(A, axis=0))

            V = get_eigenvectors_matrix(A, 2)  # Not normalized
            Vp = pinv(V)
            C = np.dot(M_0, Vp)
            CV = np.dot(C, V)
            M = (CV.T / (np.sum(CV, axis=1))).T

            for j in range(len(omega1_array)):

                omega1 = omega1_array[j]
                omega2 = -n1/n2*omega1
                omega = np.array(n1 * [omega1] + n2 * [omega2])

                # Integrate complete dynamics

                theta0 = np.arccos(1/np.arange(1, N+1))
                # print(theta0)
                # 2*np.pi*np.random.rand(N)
                z0 = np.exp(1j * theta0)
                args_kuramoto = (omega, sigma)
                kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                                  "dop853", theta0,
                                                  *args_kuramoto)

                r1 = np.absolute(
                    np.sum(M[0, :] * np.exp(1j * kuramoto_sol),
                           axis=1))
                r2 = np.absolute(
                    np.sum(M[1, :] * np.exp(1j * kuramoto_sol),
                           axis=1))
                r = np.absolute(
                    np.sum((n1*M[0, :] + n2*M[1, :])*np.exp(1j * kuramoto_sol),
                           axis=1)) / N
                r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                                 ) / len(r1[5 * int(t1 // dt) // 10:])
                r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                                 ) / len(r2[5 * int(t1 // dt) // 10:])
                r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                                ) / len(r[5 * int(t1 // dt) // 10:])

                # r1_uni = np.absolute(
                #     np.sum(M_0[0, :] * np.exp(1j * kuramoto_sol),
                #            axis=1))
                # r2_uni = np.absolute(
                #     np.sum(M_0[1, :] * np.exp(1j * kuramoto_sol),
                #            axis=1))
                # r_uni = np.absolute(
                #     np.sum((n1 * M_0[0, :] + n2 * M_0[1, :]) *
                #            np.exp(1j * kuramoto_sol), axis=1)) / N
                # r1_uni_mean = np.sum(r1_uni[5 * int(t1 // dt) // 10:]
                #                     ) / len(r1_uni[5 * int(t1 // dt) // 10:])
                # r2_uni_mean = np.sum(r2_uni[5 * int(t1 // dt) // 10:]
                #                     ) / len(r2_uni[5 * int(t1 // dt) // 10:])
                # r_uni_mean = np.sum(r_uni[5 * int(t1 // dt) // 10:]
                #                     ) / len(r_uni[5 * int(t1 // dt) // 10:])

                # Integrate reduced dynamics

                Z0 = np.dot(M, z0)

                # Spectral weights
                alpha = np.sum(np.dot(M, A), axis=1)
                MAMp = multi_dot([M, A, pinv(M)])
                MKMp = multi_dot([M, K, pinv(M)])  # np.diag(alpha)  #
                MWMp = multi_dot([M, np.diag(omega), pinv(M)])
                # from scipy.linalg import eig
                # print(eig(A)[0], eig(MAMp)[0],
                #       eig(multi_dot([M_0, A, pinv(M_0)]))[0])

                args_red_kuramoto_2D = (MWMp, sigma, N, MKMp, alpha)
                red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                      reduced_kuramoto_2D2,
                                                      MAMp, "zvode", Z0,
                                                      *args_red_kuramoto_2D)
                Z1, Z2 =\
                    red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
                R1, R2 =\
                    np.absolute(Z1), np.absolute(Z2)
                R = np.absolute(n1*Z1 + n2*Z2)/N
                R1_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 )/len(R1[5*int(t1//dt)//10:])
                R2_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 )/len(R2[5*int(t1//dt)//10:])
                R_mean = np.sum(R[5*int(t1//dt)//10:]
                                )/len(R[5*int(t1//dt)//10:])

                # Uniform weights (weights for the frequency reduction)
                # Z00 = np.dot(M_0, z0)
                # alpha0 = np.sum(np.dot(M_0, A), axis=1)
                # MAMp0 = multi_dot([M_0, A, pinv(M_0)])
                # MKMp0 = multi_dot([M_0, K, pinv(M_0)])  # np.diag(alpha0) #
                # MWMp0 = multi_dot([M_0, np.diag(omega), pinv(M_0)])
                #
                # args_red_kuramoto_2D = (MWMp0, sigma, N, MKMp0, alpha0)
                # red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                #                                       reduced_kuramoto_2D2,
                #                                       MAMp0, "zvode", Z00,
                #                                       *args_red_kuramoto_2D)
                # Z1_uni, Z2_uni = \
                #     red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
                # R1_uni, R2_uni = \
                #     np.absolute(Z1_uni), np.absolute(Z2_uni)
                # R_uni = np.absolute(n1 * Z1_uni + n2 * Z2_uni) / N
                # R1_uni_mean = np.sum(R1_uni[5 * int(t1 // dt) // 10:]
                #                     ) / len(R1_uni[5 * int(t1 // dt) // 10:])
                # R2_uni_mean = np.sum(R2_uni[5 * int(t1 // dt) // 10:]
                #                     ) / len(R2_uni[5 * int(t1 // dt) // 10:])
                # R_uni_mean = np.sum(R_uni[5 * int(t1 // dt) // 10:]
                #                     ) / len(R_uni[5 * int(t1 // dt) // 10:])

                r_matrix[i, j] = r_mean
                r1_matrix[i, j] = r1_mean
                r2_matrix[i, j] = r2_mean

                # r_uni_matrix[i, j] = r_uni_mean
                # r1_uni_matrix[i, j] = r1_uni_mean
                # r2_uni_matrix[i, j] = r2_uni_mean

                R_matrix[i, j] = R_mean
                R1_matrix[i, j] = R1_mean
                R2_matrix[i, j] = R2_mean

                # R_uni_matrix[i, j] = R_uni_mean
                # R1_uni_matrix[i, j] = R1_uni_mean
                # R2_uni_matrix[i, j] = R2_uni_mean

                if plot_temporal_series:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(9, 6))

                    first_community_color = "#2171b5"
                    reduced_first_community_color = "#9ecae1"
                    second_community_color = "#f16913"
                    reduced_second_community_color = "#fdd0a2"
                    ylim = [-0.02, 1.1]

                    plt.subplot(311)
                    plt.suptitle("$p = {}, \\omega_1 = {},"
                                 " \\omega_2 = {}$"
                                 .format(np.round(p_out_array[i], 3),
                                         np.round(omega1_array[j], 3),
                                         np.round(-n1/n2*omega1, 3)), y=1.0)
                    plt.plot(r, color="k", label="Complete spectral")
                    plt.plot(R, color="grey", label="Reduced spectral")
                    plt.plot(r_mean*np.ones(int(t1//dt)), color="r")
                    plt.plot(R_mean*np.ones(int(t1//dt)), color="orange")
                    plt.ylim(ylim)
                    plt.ylabel("$R$", fontsize=12)
                    plt.legend(loc=1, fontsize=10)

                    # plt.subplot(322)
                    # plt.plot(r_uni, color="k", label="Complete uniform")
                    # plt.plot(R_uni, color="grey", label="Reduced uniform")
                    # plt.plot(r_uni_mean*np.ones(int(t1//dt)), color="r")
                    # plt.plot(R_uni_mean*np.ones(int(t1//dt)), color="orange")
                    # # plt.plot(r_mean * np.ones(len(r)), color="r")
                    # plt.ylim(ylim)
                    # plt.legend(loc=1, fontsize=10)

                    plt.subplot(312)
                    plt.plot(r1, color=first_community_color)
                    plt.plot(R1, color=reduced_first_community_color)
                    plt.ylabel("$R_1$", fontsize=12)
                    plt.ylim(ylim)

                    # plt.subplot(324)
                    # plt.plot(r1_uni, color=first_community_color)
                    # plt.plot(R1_uni, color=reduced_first_community_color)
                    # plt.ylim(ylim)

                    plt.subplot(313)
                    plt.plot(r2, color=second_community_color)
                    plt.plot(R2, color=reduced_second_community_color)
                    plt.ylabel("$R_2$", fontsize=12)
                    plt.xlabel("$t$", fontsize=12)
                    plt.ylim(ylim)

                    # plt.subplot(316)
                    # plt.plot(r2_uni, color=second_community_color)
                    # plt.plot(R2_uni, color=reduced_second_community_color)
                    # plt.ylim(ylim)
                    # plt.xlabel("$t$", fontsize=12)

                    plt.tight_layout()

                    plt.show()

        R_dictionary["r{}".format(N)] = r_matrix.tolist()
        R_dictionary["r1{}".format(N)] = r1_matrix.tolist()
        R_dictionary["r2{}".format(N)] = r2_matrix.tolist()

        # R_dictionary["r_uni{}".format(N)] = r_uni_matrix.tolist()
        # R_dictionary["r1_uni{}".format(N)] = r1_uni_matrix.tolist()
        # R_dictionary["r2_uni{}".format(N)] = r2_uni_matrix.tolist()

        R_dictionary["R{}".format(N)] = R_matrix.tolist()
        R_dictionary["R1{}".format(N)] = R1_matrix.tolist()
        R_dictionary["R2{}".format(N)] = R2_matrix.tolist()

        # R_dictionary["R_uni{}".format(N)] = R_uni_matrix.tolist()
        # R_dictionary["R1_uni{}".format(N)] = R1_uni_matrix.tolist()
        # R_dictionary["R2_uni{}".format(N)] = R2_uni_matrix.tolist()

    R_dictionary["p_out_array"] = p_out_array.tolist()
    R_dictionary["omega1_array"] = omega1_array.tolist()
    R_dictionary["N_array"] = N_array.tolist()

    return R_dictionary


# Old reduction


def get_uniform_data_kuramoto_2D_two_triangles(sigma_array, omega1_array,
                                               t0, t1, dt,
                                               plot_temporal_series=0,
                                               plot_temporal_series_2=0):
    """
    OLD CODE See get_data_kuramoto_two_triangles
    Generate data for different reduced Kuramoto dynamics on SBM graphs
    when one module has one natural frequency and the other has
    another natural frequency (omega2 = -n1/n2*omega1).

    :param sigma_array: Array of the coupling constant sigma
    :param omega1_array: Array of the natural frequencies of the first
                         community
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :param plot_temporal_series_2: (boolean)
                                  Show temporal series (1) or not (0)
                                  of r1, R1, r2, R2, phi = phi1 - phi2,
                                  Phi = Phi1 - Phi2 when the weights are
                                  uniform in M
    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [[--- r ---]],
                               "r1",            [[--- r1---]],
                               "r2",            [[--- r2---]],
                               "r_uni",         [[--- r_uni ---]],
                               "r1_uni",        [[--- r1_uni ---]],
                               "r2_uni",        [[--- r2_uni ---]],
                               ...                ...
                               "R_none",        [[--- R_none ---]],
                               ...                ...
                               "R_all",         [[--- R_all ---]],
                               ...                ...
                               "R_hatredA",     [[--- R_hatredA ---]],
                               ...                ...
                               "R_hatLambda",   [[--- R_hatLambda ---]],
                               ...                ...
                               "R_uni",         [[--- R_uni ---]],
                               ...                ...
                               "sigma_array",    sigma_array,
                               "omega1_array",   omega1_array}
            where the values [[---X---]] is an array of shape
            len(sigma_list) times len(omega_1_list) of the order parameter X.
    """
    R_dictionary = {}
    r_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r1_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r2_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    r_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r1_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    r2_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_none_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_none_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_none_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_all_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_all_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_all_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_hatredA_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_hatredA_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_hatredA_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_hatLambda_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_hatLambda_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_hatLambda_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    R_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R1_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))
    R2_uni_matrix = np.zeros((len(sigma_array), len(omega1_array)))

    n1, n2 = 3, 3
    N = n1 + n2

    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)
        sigma = sigma_array[i]
        A = np.array([[0, 1, 1, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [1, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 1, 0]])
        K = np.diag(np.sum(A, axis=0))
        P = (np.block([[np.ones(n1), np.zeros(n2)],
                       [np.zeros(n1), np.ones(n2)]])).T

        # C = np.array([[0.4142, -0.3298],
        #               [0.4142,  0.3298]])
        # V = get_eigenvectors_matrix(A, 2)  # Not normalized
        # # Vp = pinv(V)
        # # C = np.dot(M_0, Vp)
        # # CV = np.dot(C, V)
        # CV = np.dot(C, V)
        # M = (CV.T / (np.sum(CV, axis=1))).T
        # print(M)
        M = np.array([[0.2929, 0.2929, 0.3143, 0.0999, 0, 0],
                      [0, 0, 0.0999, 0.3143, 0.2929, 0.2929]])
        M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                        [np.zeros(n1), 1 / n2 * np.ones(n2)]])
        # (CV.T / (np.sum(CV, axis=1))).T
        for j in range(len(omega1_array)):

            omega1 = omega1_array[j]
            omega2 = -n1/n2*omega1
            omega = np.array(n1 * [omega1] + n2 * [omega2])
            omega_array = np.array([omega1, omega2])

            # Integrate complete dynamics

            theta0 = 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            args_kuramoto = (omega, sigma)
            kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                              "dop853", theta0,
                                              *args_kuramoto)

            r1 = np.absolute(
                np.sum(M[0, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r = np.absolute(
                np.sum((n1*M[0, :] + n2*M[1, :])*np.exp(1j * kuramoto_sol),
                       axis=1)) / N
            r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
            r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

            r1_uni = np.absolute(
                np.sum(M_0[0, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r2_uni = np.absolute(
                np.sum(M_0[1, :] * np.exp(1j * kuramoto_sol),
                       axis=1))
            r_uni = np.absolute(
                np.sum((n1 * M_0[0, :] + n2 * M_0[1, :]) *
                       np.exp(1j * kuramoto_sol), axis=1)) / N
            r1_uni_mean = np.sum(r1_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_uni_mean = np.sum(r2_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r2[5 * int(t1 // dt) // 10:])
            r_uni_mean = np.sum(r_uni[5 * int(t1 // dt) // 10:]
                                ) / len(r[5 * int(t1 // dt) // 10:])

            # Integrate reduced dynamics

            Z0 = np.dot(M, z0)
            redA = multi_dot([M, A, P])
            hatredA = (multi_dot([M**2, A, P]).T / np.diag(np.dot(M, M.T))).T
            hatLambda = multi_dot([M, A, pinv(M)])

            # print("\n", "\n", "redA = MAP =", "\n", redA, "\n", "\n",
            #       "hatredA = MKM+ =", "\n", hatredA, "\n", "\n",
            #       "hatLambda = MAM+", "\n", hatLambda)

            #  perturbation == "None":
            args_red_kuramoto_2D = (omega_array, sigma, N,
                                    redA, redA)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D,
                                                  redA, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1_none, Z2_none = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1_none, R2_none = np.absolute(Z1_none), np.absolute(Z2_none)

            R_none = np.absolute(n1 * Z1_none + n2 * Z2_none) / N
            R1_none_mean = np.sum(R1_none[5 * int(t1 // dt) // 10:]
                                  ) / len(R1_none[5 * int(t1 // dt) // 10:])
            R2_none_mean = np.sum(R2_none[5 * int(t1 // dt) // 10:]
                                  ) / len(R2_none[5 * int(t1 // dt) // 10:])
            R_none_mean = np.sum(R_none[5 * int(t1 // dt) // 10:]
                                 ) / len(R_none[5 * int(t1 // dt) // 10:])

            # perturbation == "All":
            args_red_kuramoto_2D = (omega_array, sigma, N,
                                    hatredA, hatLambda)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D,
                                                  redA, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1_all, Z2_all = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1_all, R2_all = np.absolute(Z1_all), np.absolute(Z2_all)
            R_all = np.absolute(n1 * Z1_all + n2 * Z2_all) / N
            R1_all_mean = np.sum(R1_all[5 * int(t1 // dt) // 10:]
                                 ) / len(R1_all[5 * int(t1 // dt) // 10:])
            R2_all_mean = np.sum(R2_all[5 * int(t1 // dt) // 10:]
                                 ) / len(R2_all[5 * int(t1 // dt) // 10:])
            R_all_mean = np.sum(R_all[5 * int(t1 // dt) // 10:]
                                ) / len(R_all[5 * int(t1 // dt) // 10:])

            # perturbation == "hatredA":
            args_red_kuramoto_2D = (omega_array, sigma, N,
                                    hatredA, redA)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D,
                                                  redA, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_hatredA_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                     ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_hatredA_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                     ) / len(R2[5 * int(t1 // dt) // 10:])
            R_hatredA_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                    ) / len(R[5 * int(t1 // dt) // 10:])

            # perturbation == "hatLambda":
            args_red_kuramoto_2D = (omega_array, sigma, N,
                                    redA, hatLambda)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D,
                                                  redA, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1_hatLambda, Z2_hatLambda =\
                red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1_hatLambda, R2_hatLambda =\
                np.absolute(Z1_hatLambda), np.absolute(Z2_hatLambda)
            R_hatLambda = np.absolute(n1*Z1_hatLambda + n2*Z2_hatLambda)/N
            R1_hatLambda_mean = np.sum(R1_hatLambda[5 * int(t1 // dt) // 10:]
                                       )/len(R1_hatLambda[5*int(t1//dt)//10:])
            R2_hatLambda_mean = np.sum(R2_hatLambda[5 * int(t1 // dt) // 10:]
                                       )/len(R2_hatLambda[5*int(t1//dt)//10:])
            R_hatLambda_mean = np.sum(R_hatLambda[5*int(t1//dt)//10:]
                                      )/len(R_hatLambda[5*int(t1//dt)//10:])

            # old reduction
            alpha = np.sum(np.dot(M, A), axis=1)
            MAMp = multi_dot([M, A, pinv(M)])
            # from scipy.linalg import eig
            # print(eig(A)[0], eig(MAMp)[0],
            #       eig(multi_dot([M_0, A, pinv(M_0)]))[0])
            MKMp = multi_dot([M, K, pinv(M)])  # np.diag(alpha)  #
            MWMp = multi_dot([M, np.diag(omega), pinv(M)])

            args_red_kuramoto_2D = (MWMp, sigma, N, MKMp, alpha)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D2,
                                                  MAMp, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1_old, Z2_old =\
                red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1_old, R2_old =\
                np.absolute(Z1_old), np.absolute(Z2_old)
            R_old = np.absolute(n1*Z1_old + n2*Z2_old)/N
            # R1_old_mean = np.sum(R1_old[5 * int(t1 // dt) // 10:]
            #                      )/len(R1_old[5*int(t1//dt)//10:])
            # R2_old_mean = np.sum(R2_old[5 * int(t1 // dt) // 10:]
            #                      )/len(R2_old[5*int(t1//dt)//10:])
            # R_old_mean = np.sum(R_old[5*int(t1//dt)//10:]
            #                     )/len(R_old[5*int(t1//dt)//10:])

            # Uniform, perturbation == "None" and M = M_0 = M_T:
            Z0 = np.dot(M_0, z0)
            redA0 = multi_dot([M_0, A, P])
            args_red_kuramoto_2D = (omega_array, sigma, N,
                                    redA0, redA0)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D,
                                                  redA0, "zvode", Z0,
                                                  *args_red_kuramoto_2D)

            Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_uni_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_uni_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 ) / len(R2[5 * int(t1 // dt) // 10:])
            R_uni_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                ) / len(R[5 * int(t1 // dt) // 10:])

            r_matrix[i, j] = r_mean
            r1_matrix[i, j] = r1_mean
            r2_matrix[i, j] = r2_mean

            r_uni_matrix[i, j] = r_uni_mean
            r1_uni_matrix[i, j] = r1_uni_mean
            r2_uni_matrix[i, j] = r2_uni_mean

            R_none_matrix[i, j] = R_none_mean
            R1_none_matrix[i, j] = R1_none_mean
            R2_none_matrix[i, j] = R2_none_mean

            R_all_matrix[i, j] = R_all_mean
            R1_all_matrix[i, j] = R1_all_mean
            R2_all_matrix[i, j] = R2_all_mean

            R_hatredA_matrix[i, j] = R_hatredA_mean
            R1_hatredA_matrix[i, j] = R1_hatredA_mean
            R2_hatredA_matrix[i, j] = R2_hatredA_mean

            R_hatLambda_matrix[i, j] = R_hatLambda_mean
            R1_hatLambda_matrix[i, j] = R1_hatLambda_mean
            R2_hatLambda_matrix[i, j] = R2_hatLambda_mean

            R_uni_matrix[i, j] = R_uni_mean
            R1_uni_matrix[i, j] = R1_uni_mean
            R2_uni_matrix[i, j] = R2_uni_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(6, 7))

                plt.subplot(311)
                plt.suptitle("$\\sigma = {}, \\omega_1 = {}, \\omega_2 = {}$"
                             .format(np.round(sigma_array[i], 3),
                                     np.round(omega1_array[j], 3),
                                     np.round(-n1/n2*omega1, 3)), y=1.0)
                plt.plot(r, color="grey", label="Complete")
                plt.plot(r_uni, color="k", label="Complete uni")
                plt.plot(R, color="purple", label="Uniform")
                plt.plot(R_none, color="b", label="None")
                plt.plot(R_hatLambda, color="r", label="hatLambda")
                plt.plot(R_all, color="orange", label="All")
                plt.plot(R_old, color="brown", label="Old")
                # plt.plot(r_mean * np.ones(len(r)), color="r")
                plt.ylabel("$R$", fontsize=12)
                plt.ylim([-0.02, 1.02])
                plt.legend(loc=1, fontsize=10)

                plt.subplot(312)
                plt.plot(r1, color="grey")
                plt.plot(r1_uni, color="k")
                plt.plot(R1, color="purple")
                plt.plot(R1_none, color="b")
                plt.plot(R1_hatLambda, color="r")
                plt.plot(R1_all, color="orange")
                plt.plot(R1_old, color="brown")
                plt.ylabel("$R_1$", fontsize=12)
                plt.ylim([-0.02, 1.02])

                plt.subplot(313)
                plt.plot(r2, color="grey")
                plt.plot(r2_uni, color="k")
                plt.plot(R2, color="purple")
                plt.plot(R2_none, color="b")
                plt.plot(R2_hatLambda, color="r")
                plt.plot(R2_all, color="orange")
                plt.plot(R2_old, color="brown")
                plt.ylabel("$R_2$", fontsize=12)
                plt.ylim([-0.02, 1.02])
                plt.xlabel("$t$", fontsize=12)

                plt.tight_layout()

                plt.show()

            if plot_temporal_series_2:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 10))

                Phi = np.angle(Z1) - np.angle(Z2)

                phi1_uni = np.angle(
                    np.sum(M_0[0, 0:n1] * np.exp(1j * kuramoto_sol[:, 0:n1]),
                           axis=1))
                phi2_uni = np.angle(
                    np.sum(M_0[1, n1:] * np.exp(1j * kuramoto_sol[:, n1:]),
                           axis=1))
                plt.subplot(311)
                plt.plot(r1_uni, color="k")
                plt.plot(R1, color="b", linestyle='-')
                plt.ylabel("$R_1$")

                plt.subplot(312)
                plt.plot(r2_uni, color="k")
                plt.plot(R2, color="r", linestyle='-')
                plt.ylabel("$R_2$")

                plt.subplot(313)
                # plt.scatter(t0, Phi[0], color="k", s=50)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1),
                            phi1_uni - phi2_uni,
                            color="k", s=10)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1), Phi,
                            color="purple", s=5)
                plt.ylabel("$\\Phi = \\Phi_1 - \\Phi_2$")
                plt.xlabel("Time $t$")
                plt.show()

    R_dictionary["r"] = r_matrix.tolist()
    R_dictionary["r1"] = r1_matrix.tolist()
    R_dictionary["r2"] = r2_matrix.tolist()

    R_dictionary["r_uni"] = r_uni_matrix.tolist()
    R_dictionary["r1_uni"] = r1_uni_matrix.tolist()
    R_dictionary["r2_uni"] = r2_uni_matrix.tolist()

    R_dictionary["R_none"] = R_none_matrix.tolist()
    R_dictionary["R1_none"] = R1_none_matrix.tolist()
    R_dictionary["R2_none"] = R2_none_matrix.tolist()

    R_dictionary["R_all"] = R_all_matrix.tolist()
    R_dictionary["R1_all"] = R1_all_matrix.tolist()
    R_dictionary["R2_all"] = R2_all_matrix.tolist()

    R_dictionary["R_hatredA"] = R_hatredA_matrix.tolist()
    R_dictionary["R1_hatredA"] = R1_hatredA_matrix.tolist()
    R_dictionary["R2_hatredA"] = R2_hatredA_matrix.tolist()

    R_dictionary["R_hatLambda"] = R_hatLambda_matrix.tolist()
    R_dictionary["R1_hatLambda"] = R1_hatLambda_matrix.tolist()
    R_dictionary["R2_hatLambda"] = R2_hatLambda_matrix.tolist()

    R_dictionary["R_uni"] = R_uni_matrix.tolist()
    R_dictionary["R1_uni"] = R1_uni_matrix.tolist()
    R_dictionary["R2_uni"] = R2_uni_matrix.tolist()

    R_dictionary["sigma_array"] = sigma_array.tolist()
    R_dictionary["omega1_array"] = omega1_array.tolist()

    return R_dictionary


def get_R_matrix_from_R_dictionary(keys_array, R_dictionary):
    """
    :param keys_array: Ordered keys (order must be as in the description of
                                     R_dictionary)
    :param R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [[--- r ---]],
                               "r1",            [[--- r1---]],
                               "r2",            [[--- r2---]],
                               "r_uni",         [[--- r_uni ---]],
                               "r1_uni",        [[--- r1_uni ---]],
                               "r2_uni",        [[--- r2_uni ---]],
                               "R_none",        [[--- R_none ---]],
                               ...                ...
                               "R_all",         [[--- R_all ---]],
                               ...                ...
                               "R_hatredA",     [[--- R_hatredA ---]],
                               ...                ...
                               "R_hatLambda",   [[--- R_hatLambda ---]],
                               ...                ...
                               "R_uni",         [[--- R_uni ---]],
                               ...                ...
                               "p_out_array",    p_out_array,
                               "omega1_array",   omega1_array}
            where the values [[---X---]] is an array of shape
            len(p_out_list) times len(omega_1_list) of the order parameter X.
    :return:
    """
    R_matrix = np.zeros((len(keys_array),
                         len(R_dictionary["sigma_array"]) *
                         len(R_dictionary["omega1_array"])))
    for i in range(len(keys_array)):
        R_matrix[i, :] = np.array(R_dictionary[keys_array[i]]).flatten()

    return R_matrix


def get_RMSE_errors(R_matrix):
    """

    :param R_matrix is a 21 by nb_data matrix of the form
            [[--- r ---],
             [--- r1 ---],
             [--- r2 ---],
             ...
             [--- r_uni ---],
             ...
             [--- R_none ---],
             ...
             [--- R_all ---],
             ...
             [--- R_hatredA ---],
             ...
             [--- R_hatLambda ---],
             ...
             [--- R_uni ---]]
    :return:
    """
    RMSE_list, RMSE1_list, RMSE2_list = [], [], []
    j = 6
    for i in range(5):
        if i < 4:
            RMSE_list.append(RMSE(R_matrix[0], R_matrix[j]))
            RMSE1_list.append(RMSE(R_matrix[1], R_matrix[j+1]))
            RMSE2_list.append(RMSE(R_matrix[2], R_matrix[j+2]))
        else:
            RMSE_list.append(RMSE(R_matrix[3], R_matrix[18]))
            RMSE1_list.append(RMSE(R_matrix[4], R_matrix[19]))
            RMSE2_list.append(RMSE(R_matrix[5], R_matrix[20]))
        j += 3
    return RMSE_list, RMSE1_list, RMSE2_list


def get_mean_L1_errors(R_matrix):
    """

    :param R_matrix is a 21 by nb_data matrix of the form
            [[--- r ---],
             [--- r1 ---],
             [--- r2 ---],
             ...
             [--- r_uni ---],
             ...
             [--- R_none ---],
             ...
             [--- R_all ---],
             ...
             [--- R_hatredA ---],
             ...
             [--- R_hatLambda ---],
             ...
             [--- R_uni ---]]
    :return:
    """
    mean_L1_list, mean_L1_1_list, mean_L1_2_list = [], [], []
    j = 6
    for i in range(5):
        if i < 4:
            mean_L1_list.append(mean_L1(R_matrix[0], R_matrix[j]))
            mean_L1_1_list.append(mean_L1(R_matrix[1], R_matrix[j+1]))
            mean_L1_2_list.append(mean_L1(R_matrix[2], R_matrix[j+2]))
        else:
            mean_L1_list.append(mean_L1(R_matrix[3], R_matrix[18]))
            mean_L1_1_list.append(mean_L1(R_matrix[4], R_matrix[19]))
            mean_L1_2_list.append(mean_L1(R_matrix[5], R_matrix[20]))
        j += 3
    return mean_L1_list, mean_L1_1_list, mean_L1_2_list


def get_L1_errors(R_matrix):
    L1_matrix = np.zeros(np.shape(R_matrix))
    L1_1_matrix = np.zeros(np.shape(R_matrix))
    L1_2_matrix = np.zeros(np.shape(R_matrix))
    j = 6
    for i in range(5):
        if i < 4:
            L1_matrix[i] = L1(R_matrix[0], R_matrix[j])
            L1_1_matrix[i] = L1(R_matrix[1], R_matrix[j+1])
            L1_2_matrix[i] = L1(R_matrix[2], R_matrix[j+2])
        else:
            L1_matrix[i] = L1(R_matrix[3], R_matrix[18])
            L1_1_matrix[i] = L1(R_matrix[4], R_matrix[19])
            L1_2_matrix[i] = L1(R_matrix[5], R_matrix[20])
        j += 3
    return L1_matrix, L1_1_matrix, L1_2_matrix


def get_marginal_L1_errors(axis, nb_omega1, nb_p_out,
                           L1_matrix, L1_1_matrix, L1_2_matrix):
    if axis == 0:
        nb_data_parameter = nb_omega1
    else:
        nb_data_parameter = nb_p_out

    L1_marginal_matrix = np.zeros((len(L1_matrix[:, 0]),
                                   nb_data_parameter))
    L1_1_marginal_matrix = np.zeros((len(L1_1_matrix[:, 0]),
                                     nb_data_parameter))
    L1_2_marginal_matrix = np.zeros((len(L1_2_matrix[:, 0]),
                                     nb_data_parameter))
    for i in range(len(L1_matrix[:, 0])):
        L1_matrix_perturbation = np.reshape(L1_matrix[i, :],
                                            (nb_p_out, nb_omega1))
        L1_1_matrix_perturbation = np.reshape(L1_1_matrix[i, :],
                                              (nb_p_out, nb_omega1))
        L1_2_matrix_perturbation = np.reshape(L1_2_matrix[i, :],
                                              (nb_p_out, nb_omega1))
        # for j in range(nb_omega1*nb_p_out):
        L1_marginal_matrix[i, :] = np.mean(L1_matrix_perturbation,
                                           axis=axis)
        L1_1_marginal_matrix[i, :] = np.mean(L1_1_matrix_perturbation,
                                             axis=axis)
        L1_2_marginal_matrix[i, :] = np.mean(L1_2_matrix_perturbation,
                                             axis=axis)
    return L1_marginal_matrix, L1_1_marginal_matrix, L1_2_marginal_matrix


def get_uniform_data_kuramoto_2D(p_out_array, omega1_array, sizes, t0, t1, dt,
                                 sigma, plot_temporal_series=0,
                                 plot_temporal_series_2=0):
    """
    Generate data for different reduced Kuramoto dynamics on SBM graphs
    when one module has one natural frequency and the other has
    another natural frequency (omega2 = -n1/n2*omega1).

    :param p_out_array: Array of the probability to be connected to the other
                        layer of oscillator
    :param omega1_array: Array of the natural frequencies of the first
                         community
    :param sizes: [n1, n2] Community sizes
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param sigma: Coupling constant
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :param plot_temporal_series_2: (boolean)
                                  Show temporal series (1) or not (0)
                                  of r1, R1, r2, R2, phi = phi1 - phi2,
                                  Phi = Phi1 - Phi2 when the weights are
                                  uniform in M
    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [[--- r ---]],
                               "r1",            [[--- r1---]],
                               "r2",            [[--- r2---]],
                               "r_uni",         [[--- r_uni ---]],
                               "r1_uni",        [[--- r1_uni ---]],
                               "r2_uni",        [[--- r2_uni ---]],
                               ...                ...
                               "R_none",        [[--- R_none ---]],
                               ...                ...
                               "R_all",         [[--- R_all ---]],
                               ...                ...
                               "R_hatredA",     [[--- R_hatredA ---]],
                               ...                ...
                               "R_hatLambda",   [[--- R_hatLambda ---]],
                               ...                ...
                               "R_uni",         [[--- R_uni ---]],
                               ...                ...
                               "p_out_array",    p_out_array,
                               "omega1_array",   omega1_array}
            where the values [[---X---]] is an array of shape
            len(p_out_list) times len(omega_1_list) of the order parameter X.
    """
    n1, n2 = sizes
    N = sum(sizes)
    R_dictionary = {}
    r_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r1_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r2_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    r_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r1_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r2_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_none_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_none_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_none_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_all_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_all_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_all_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_hatredA_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_hatredA_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_hatredA_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_hatLambda_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_hatLambda_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_hatLambda_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    for i in tqdm(range(len(p_out_array))):
        time.sleep(1)

        pq = [[0, p_out_array[i]], [p_out_array[i], 0]]
        P = (np.block([[np.ones(n1), np.zeros(n2)],
                       [np.zeros(n1), np.ones(n2)]])).T
        A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
        V = get_eigenvectors_matrix(A, 2)  # Not normalized
        Vp = pinv(V)
        # M_T = np.block([[V[0, 0:n1]/np.sum(V[0, 0:n1]), np.zeros(n2)],
        #                 [np.zeros(n1), V[0, n1:]/np.sum(V[0, n1:])]])

        M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                        [np.zeros(n1), 1 / n2 * np.ones(n2)]])
        C = np.dot(M_0, Vp)
        CV = np.dot(C, V)
        M = (CV.T / (np.sum(CV, axis=1))).T

        for j in range(len(omega1_array)):

            omega1 = omega1_array[j]
            omega2 = -n1/n2*omega1
            omega = np.array(n1 * [omega1] + n2 * [omega2])
            omega_array = np.array([omega1, omega2])

            # Integrate complete dynamics

            theta0 = 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            args_kuramoto = (omega, sigma)
            kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                              "dop853", theta0,
                                              *args_kuramoto)

            r1 = np.absolute(
                np.sum(M[0, 0:n1] * np.exp(1j * kuramoto_sol[:, 0:n1]),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, n1:] * np.exp(1j * kuramoto_sol[:, n1:]),
                       axis=1))
            r = np.absolute(
                np.sum((n1*M[0, :] + n2*M[1, :])*np.exp(1j * kuramoto_sol),
                       axis=1)) / N
            r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
            r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

            r1_uni = np.absolute(
                np.sum(M_0[0, 0:n1] * np.exp(1j * kuramoto_sol[:, 0:n1]),
                       axis=1))
            r2_uni = np.absolute(
                np.sum(M_0[1, n1:] * np.exp(1j * kuramoto_sol[:, n1:]),
                       axis=1))
            r_uni = np.absolute(
                np.sum((n1 * M_0[0, :] + n2 * M_0[1, :]) *
                       np.exp(1j * kuramoto_sol), axis=1)) / N
            r1_uni_mean = np.sum(r1_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_uni_mean = np.sum(r2_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r2[5 * int(t1 // dt) // 10:])
            r_uni_mean = np.sum(r_uni[5 * int(t1 // dt) // 10:]
                                ) / len(r[5 * int(t1 // dt) // 10:])

            # Integrate recuced dynamics

            Z0 = np.dot(M, z0)
            redA = multi_dot([M, A, P])
            hatredA = (multi_dot([M**2, A, P]).T / np.diag(np.dot(M, M.T))).T
            hatLambda = multi_dot([M, A, pinv(M)])

            #  perturbation == "None":
            args_red_kuramoto_2D = (omega_array, sigma, N,
                                    redA, redA)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D,
                                                  redA, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)

            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_none_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                  ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_none_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                  ) / len(R2[5 * int(t1 // dt) // 10:])
            R_none_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                 ) / len(R[5 * int(t1 // dt) // 10:])

            # perturbation == "All":
            args_red_kuramoto_2D = (omega_array, sigma, N,
                                    hatredA, hatLambda)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D,
                                                  redA, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_all_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_all_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 ) / len(R2[5 * int(t1 // dt) // 10:])
            R_all_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                ) / len(R[5 * int(t1 // dt) // 10:])

            # perturbation == "hatredA":
            args_red_kuramoto_2D = (omega_array, sigma, N,
                                    hatredA, redA)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D,
                                                  redA, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_hatredA_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                     ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_hatredA_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                     ) / len(R2[5 * int(t1 // dt) // 10:])
            R_hatredA_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                    ) / len(R[5 * int(t1 // dt) // 10:])

            # perturbation == "hatLambda":
            args_red_kuramoto_2D = (omega_array, sigma, N,
                                    redA, hatLambda)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D,
                                                  redA, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_hatLambda_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                       ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_hatLambda_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                       ) / len(R2[5 * int(t1 // dt) // 10:])
            R_hatLambda_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                      ) / len(R[5 * int(t1 // dt) // 10:])

            # Uniform, perturbation == "None" and M = M_0 = M_T:
            Z0 = np.dot(M_0, z0)
            redA0 = multi_dot([M_0, A, P])
            args_red_kuramoto_2D = (omega_array, sigma, N,
                                    redA0, redA0)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D,
                                                  redA0, "zvode", Z0,
                                                  *args_red_kuramoto_2D)

            Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_uni_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_uni_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 ) / len(R2[5 * int(t1 // dt) // 10:])
            R_uni_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                ) / len(R[5 * int(t1 // dt) // 10:])

            r_matrix[i, j] = r_mean
            r1_matrix[i, j] = r1_mean
            r2_matrix[i, j] = r2_mean

            r_uni_matrix[i, j] = r_uni_mean
            r1_uni_matrix[i, j] = r1_uni_mean
            r2_uni_matrix[i, j] = r2_uni_mean

            R_none_matrix[i, j] = R_none_mean
            R1_none_matrix[i, j] = R1_none_mean
            R2_none_matrix[i, j] = R2_none_mean

            R_all_matrix[i, j] = R_all_mean
            R1_all_matrix[i, j] = R1_all_mean
            R2_all_matrix[i, j] = R2_all_mean

            R_hatredA_matrix[i, j] = R_hatredA_mean
            R1_hatredA_matrix[i, j] = R1_hatredA_mean
            R2_hatredA_matrix[i, j] = R2_hatredA_mean

            R_hatLambda_matrix[i, j] = R_hatLambda_mean
            R1_hatLambda_matrix[i, j] = R1_hatLambda_mean
            R2_hatLambda_matrix[i, j] = R2_hatLambda_mean

            R_uni_matrix[i, j] = R_uni_mean
            R1_uni_matrix[i, j] = R1_uni_mean
            R2_uni_matrix[i, j] = R2_uni_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(6, 7))

                plt.subplot(311)
                plt.suptitle("$p = {}, \\omega_1 = {}, \\omega_2 = {}$"
                             .format(np.round(p_out_array[i], 3),
                                     np.round(omega1_array[j], 3),
                                     np.round(-n1/n2*omega1, 3)), y=1.0)
                plt.plot(r_uni, color="k")
                plt.plot(R, color="g")
                # plt.plot(r_mean * np.ones(len(r)), color="r")
                plt.ylabel("$R_{uni}$", fontsize=12)
                plt.ylim([-0.02, 1.02])

                plt.subplot(312)
                plt.plot(r1_uni, color="b")
                plt.plot(R1, color="lightblue")
                plt.ylabel("$(R_1)_{uni}$", fontsize=12)
                plt.ylim([-0.02, 1.02])

                plt.subplot(313)
                plt.plot(r2_uni, color="r")
                plt.plot(R2, color="y")
                plt.ylabel("$(R_2)_{uni}$", fontsize=12)
                plt.ylim([-0.02, 1.02])
                plt.xlabel("$t$", fontsize=12)

                plt.tight_layout()

                plt.show()

            if plot_temporal_series_2:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 10))

                Phi = np.angle(Z1) - np.angle(Z2)

                phi1_uni = np.angle(
                    np.sum(M_0[0, 0:n1] * np.exp(1j * kuramoto_sol[:, 0:n1]),
                           axis=1))
                phi2_uni = np.angle(
                    np.sum(M_0[1, n1:] * np.exp(1j * kuramoto_sol[:, n1:]),
                           axis=1))
                plt.subplot(311)
                plt.plot(r1_uni, color="k")
                plt.plot(R1, color="b", linestyle='-')
                plt.ylabel("$R_1$")

                plt.subplot(312)
                plt.plot(r2_uni, color="k")
                plt.plot(R2, color="r", linestyle='-')
                plt.ylabel("$R_2$")

                plt.subplot(313)
                # plt.scatter(t0, Phi[0], color="k", s=50)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1),
                            phi1_uni - phi2_uni,
                            color="k", s=10)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1), Phi,
                            color="purple", s=5)
                plt.ylabel("$\\Phi = \\Phi_1 - \\Phi_2$")
                plt.xlabel("Time $t$")
                plt.show()

    R_dictionary["r"] = r_matrix.tolist()
    R_dictionary["r1"] = r1_matrix.tolist()
    R_dictionary["r2"] = r2_matrix.tolist()

    R_dictionary["r_uni"] = r_uni_matrix.tolist()
    R_dictionary["r1_uni"] = r1_uni_matrix.tolist()
    R_dictionary["r2_uni"] = r2_uni_matrix.tolist()

    R_dictionary["R_none"] = R_none_matrix.tolist()
    R_dictionary["R1_none"] = R1_none_matrix.tolist()
    R_dictionary["R2_none"] = R2_none_matrix.tolist()

    R_dictionary["R_all"] = R_all_matrix.tolist()
    R_dictionary["R1_all"] = R1_all_matrix.tolist()
    R_dictionary["R2_all"] = R2_all_matrix.tolist()

    R_dictionary["R_hatredA"] = R_hatredA_matrix.tolist()
    R_dictionary["R1_hatredA"] = R1_hatredA_matrix.tolist()
    R_dictionary["R2_hatredA"] = R2_hatredA_matrix.tolist()

    R_dictionary["R_hatLambda"] = R_hatLambda_matrix.tolist()
    R_dictionary["R1_hatLambda"] = R1_hatLambda_matrix.tolist()
    R_dictionary["R2_hatLambda"] = R2_hatLambda_matrix.tolist()

    R_dictionary["R_uni"] = R_uni_matrix.tolist()
    R_dictionary["R1_uni"] = R1_uni_matrix.tolist()
    R_dictionary["R2_uni"] = R2_uni_matrix.tolist()

    R_dictionary["p_out_array"] = p_out_array.tolist()
    R_dictionary["omega1_array"] = omega1_array.tolist()

    return R_dictionary


def get_uniform_data_winfree_2D(p_out_array, omega1_array, sizes, t0, t1, dt,
                                sigma, plot_temporal_series=0,
                                plot_temporal_series_2=0):
    """
    Generate data for different reduced Winfree dynamics on SBM graphs
    when one module has one natural frequency and the other has
    another natural frequency (omega2 = -n1/n2*omega1).

    :param p_out_array: Array of the probability to be connected to the other
                        layer of oscillator
    :param omega1_array: Array of the natural frequencies of the first
                         community
    :param sizes: [n1, n2] Community sizes
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param sigma: Coupling constant
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :param plot_temporal_series_2: (boolean)
                                  Show temporal series (1) or not (0)
                                  of r1, R1, r2, R2, phi = phi1 - phi2,
                                  Phi = Phi1 - Phi2 when the weights are
                                  uniform in M

    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [[--- r ---]],
                               "r1",            [[--- r1---]],
                               "r2",            [[--- r2---]],
                               "r_uni",         [[--- r_uni ---]],
                               "r1_uni",        [[--- r1_uni ---]],
                               "r2_uni",        [[--- r2_uni ---]],
                               ...                ...
                               "R_none",        [[--- R_none ---]],
                               ...                ...
                               "R_all",         [[--- R_all ---]],
                               ...                ...
                               "R_hatredA",     [[--- R_hatredA ---]],
                               ...                ...
                               "R_hatLambda",   [[--- R_hatLambda ---]],
                               ...                ...
                               "R_uni",         [[--- R_uni ---]],
                               ...                ...
                               "p_out_array",    p_out_array,
                               "omega1_array",   omega1_array}
            where the values [[---X---]] is an array of shape
            len(p_out_list) times len(omega_1_list) of the order parameter X.
    """
    n1, n2 = sizes
    N = sum(sizes)
    R_dictionary = {}
    r_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r1_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r2_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    r_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r1_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r2_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_none_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_none_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_none_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_all_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_all_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_all_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_hatredA_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_hatredA_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_hatredA_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_hatLambda_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_hatLambda_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_hatLambda_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    for i in tqdm(range(len(p_out_array))):
        time.sleep(1)

        pq = [[0, p_out_array[i]], [p_out_array[i], 0]]
        P = (np.block([[np.ones(n1), np.zeros(n2)],
                       [np.zeros(n1), np.ones(n2)]])).T
        A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
        V = get_eigenvectors_matrix(A, 2)  # Not normalized
        Vp = pinv(V)
        # M_T = np.block([[V[0, 0:n1]/np.sum(V[0, 0:n1]), np.zeros(n2)],
        #                 [np.zeros(n1), V[0, n1:]/np.sum(V[0, n1:])]])

        M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                        [np.zeros(n1), 1 / n2 * np.ones(n2)]])
        C = np.dot(M_0, Vp)
        CV = np.dot(C, V)
        M = (CV.T / (np.sum(CV, axis=1))).T

        for j in range(len(omega1_array)):

            omega1 = omega1_array[j]
            omega2 = -n1/n2*omega1
            omega = np.array(n1 * [omega1] + n2 * [omega2])
            omega_array = np.array([omega1, omega2])

            # Integrate complete dynamics

            theta0 = 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            args_winfree = (omega, sigma)
            winfree_sol = integrate_dynamics(t0, t1, dt, winfree, A,
                                             "dop853", theta0,
                                             *args_winfree)

            r1 = np.absolute(
                np.sum(M[0, 0:n1] * np.exp(1j * winfree_sol[:, 0:n1]),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, n1:] * np.exp(1j * winfree_sol[:, n1:]),
                       axis=1))
            r = np.absolute(
                np.sum((n1*M[0, :] + n2*M[1, :])*np.exp(1j * winfree_sol),
                       axis=1)) / N
            r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
            r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

            r1_uni = np.absolute(
                np.sum(M_0[0, 0:n1] * np.exp(1j * winfree_sol[:, 0:n1]),
                       axis=1))
            r2_uni = np.absolute(
                np.sum(M_0[1, n1:] * np.exp(1j * winfree_sol[:, n1:]),
                       axis=1))
            r_uni = np.absolute(
                np.sum((n1 * M_0[0, :] + n2 * M_0[1, :]) *
                       np.exp(1j * winfree_sol), axis=1)) / N
            r1_uni_mean = np.sum(r1_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_uni_mean = np.sum(r2_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r2[5 * int(t1 // dt) // 10:])
            r_uni_mean = np.sum(r_uni[5 * int(t1 // dt) // 10:]
                                ) / len(r[5 * int(t1 // dt) // 10:])

            # Integrate recuced dynamics

            Z0 = np.dot(M, z0)
            redA = multi_dot([M, A, P])
            hatredA = (multi_dot([M**2, A, P]).T / np.diag(np.dot(M, M.T))).T
            hatLambda = multi_dot([M, A, pinv(M)])

            #  perturbation == "None":
            args_red_winfree_2D = (omega_array, sigma, N,
                                   redA, redA)
            red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_winfree_2D,
                                                 redA, "zvode", Z0,
                                                 *args_red_winfree_2D)
            Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_none_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                  ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_none_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                  ) / len(R2[5 * int(t1 // dt) // 10:])
            R_none_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                 ) / len(R[5 * int(t1 // dt) // 10:])

            # perturbation == "All":
            args_red_winfree_2D = (omega_array, sigma, N,
                                   hatredA, hatLambda)
            red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_winfree_2D,
                                                 redA, "zvode", Z0,
                                                 *args_red_winfree_2D)
            Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_all_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_all_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 ) / len(R2[5 * int(t1 // dt) // 10:])
            R_all_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                ) / len(R[5 * int(t1 // dt) // 10:])

            # perturbation == "hatredA":
            args_red_winfree_2D = (omega_array, sigma, N,
                                   hatredA, redA)
            red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_winfree_2D,
                                                 redA, "zvode", Z0,
                                                 *args_red_winfree_2D)
            Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_hatredA_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                     ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_hatredA_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                     ) / len(R2[5 * int(t1 // dt) // 10:])
            R_hatredA_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                    ) / len(R[5 * int(t1 // dt) // 10:])

            # perturbation == "hatLambda":
            args_red_winfree_2D = (omega_array, sigma, N,
                                   redA, hatLambda)
            red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_winfree_2D,
                                                 redA, "zvode", Z0,
                                                 *args_red_winfree_2D)
            Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_hatLambda_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                       ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_hatLambda_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                       ) / len(R2[5 * int(t1 // dt) // 10:])
            R_hatLambda_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                      ) / len(R[5 * int(t1 // dt) // 10:])

            # Uniform, perturbation == "None" and M = M_0 = M_T:
            Z0 = np.dot(M_0, z0)
            redA0 = multi_dot([M_0, A, P])
            args_red_winfree_2D = (omega_array, sigma, N,
                                   redA0, redA0)
            red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_winfree_2D,
                                                 redA0, "zvode", Z0,
                                                 *args_red_winfree_2D)

            Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_uni_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_uni_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 ) / len(R2[5 * int(t1 // dt) // 10:])
            R_uni_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                ) / len(R[5 * int(t1 // dt) // 10:])

            r_matrix[i, j] = r_mean
            r1_matrix[i, j] = r1_mean
            r2_matrix[i, j] = r2_mean

            r_uni_matrix[i, j] = r_uni_mean
            r1_uni_matrix[i, j] = r1_uni_mean
            r2_uni_matrix[i, j] = r2_uni_mean

            R_none_matrix[i, j] = R_none_mean
            R1_none_matrix[i, j] = R1_none_mean
            R2_none_matrix[i, j] = R2_none_mean

            R_all_matrix[i, j] = R_all_mean
            R1_all_matrix[i, j] = R1_all_mean
            R2_all_matrix[i, j] = R2_all_mean

            R_hatredA_matrix[i, j] = R_hatredA_mean
            R1_hatredA_matrix[i, j] = R1_hatredA_mean
            R2_hatredA_matrix[i, j] = R2_hatredA_mean

            R_hatLambda_matrix[i, j] = R_hatLambda_mean
            R1_hatLambda_matrix[i, j] = R1_hatLambda_mean
            R2_hatLambda_matrix[i, j] = R2_hatLambda_mean

            R_uni_matrix[i, j] = R_uni_mean
            R1_uni_matrix[i, j] = R1_uni_mean
            R2_uni_matrix[i, j] = R2_uni_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(6, 7))

                plt.subplot(311)
                plt.suptitle("$p = {}, \\omega_1 = {}, \\omega_2 = {}$"
                             .format(np.round(p_out_array[i], 3),
                                     np.round(omega1_array[j], 3),
                                     np.round(-n1/n2*omega1, 3)), y=1.0)
                plt.plot(r_uni, color="k")
                plt.plot(R, color="g")
                # plt.plot(r_mean * np.ones(len(r)), color="r")
                plt.ylabel("$R_{uni}$", fontsize=12)
                plt.ylim([-0.02, 1.02])

                plt.subplot(312)
                plt.plot(r1_uni, color="b")
                plt.plot(R1, color="lightblue")
                plt.ylabel("$(R_1)_{uni}$", fontsize=12)
                plt.ylim([-0.02, 1.02])

                plt.subplot(313)
                plt.plot(r2_uni, color="r")
                plt.plot(R2, color="y")
                plt.ylabel("$(R_2)_{uni}$", fontsize=12)
                plt.ylim([-0.02, 1.02])
                plt.xlabel("$t$", fontsize=12)

                plt.tight_layout()

                plt.show()

            if plot_temporal_series_2:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 10))

                Phi = np.angle(Z1) - np.angle(Z2)

                phi1_uni = np.angle(
                    np.sum(M_0[0, 0:n1] * np.exp(1j * winfree_sol[:, 0:n1]),
                           axis=1))
                phi2_uni = np.angle(
                    np.sum(M_0[1, n1:] * np.exp(1j * winfree_sol[:, n1:]),
                           axis=1))
                plt.subplot(311)
                plt.plot(r1_uni, color="k")
                plt.plot(R1, color="b", linestyle='-')
                plt.ylabel("$R_1$")

                plt.subplot(312)
                plt.plot(r2_uni, color="k")
                plt.plot(R2, color="r", linestyle='-')
                plt.ylabel("$R_2$")

                plt.subplot(313)
                # plt.scatter(t0, Phi[0], color="k", s=50)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1),
                            phi1_uni - phi2_uni,
                            color="k", s=10)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1), Phi,
                            color="purple", s=5)
                plt.ylabel("$\\Phi = \\Phi_1 - \\Phi_2$")
                plt.xlabel("Time $t$")
                plt.show()

    R_dictionary["r"] = r_matrix.tolist()
    R_dictionary["r1"] = r1_matrix.tolist()
    R_dictionary["r2"] = r2_matrix.tolist()

    R_dictionary["r_uni"] = r_uni_matrix.tolist()
    R_dictionary["r1_uni"] = r1_uni_matrix.tolist()
    R_dictionary["r2_uni"] = r2_uni_matrix.tolist()

    R_dictionary["R_none"] = R_none_matrix.tolist()
    R_dictionary["R1_none"] = R1_none_matrix.tolist()
    R_dictionary["R2_none"] = R2_none_matrix.tolist()

    R_dictionary["R_all"] = R_all_matrix.tolist()
    R_dictionary["R1_all"] = R1_all_matrix.tolist()
    R_dictionary["R2_all"] = R2_all_matrix.tolist()

    R_dictionary["R_hatredA"] = R_hatredA_matrix.tolist()
    R_dictionary["R1_hatredA"] = R1_hatredA_matrix.tolist()
    R_dictionary["R2_hatredA"] = R2_hatredA_matrix.tolist()

    R_dictionary["R_hatLambda"] = R_hatLambda_matrix.tolist()
    R_dictionary["R1_hatLambda"] = R1_hatLambda_matrix.tolist()
    R_dictionary["R2_hatLambda"] = R2_hatLambda_matrix.tolist()

    R_dictionary["R_uni"] = R_uni_matrix.tolist()
    R_dictionary["R1_uni"] = R1_uni_matrix.tolist()
    R_dictionary["R2_uni"] = R2_uni_matrix.tolist()

    R_dictionary["p_out_array"] = p_out_array.tolist()
    R_dictionary["omega1_array"] = omega1_array.tolist()

    return R_dictionary


def get_uniform_data_cosinus_2D(p_out_array, omega1_array, sizes, t0, t1, dt,
                                sigma, plot_temporal_series=0,
                                plot_temporal_series_2=0):
    """
    Generate data for different reduced cosinus dynamics on SBM graphs
    when one module has one natural frequency and the other has
    another natural frequency (omega2 = -n1/n2*omega1).

    :param p_out_array: Array of the probability to be connected to the other
                        layer of oscillator
    :param omega1_array: Array of the natural frequencies of the first
                         community
    :param sizes: [n1, n2] Community sizes
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param sigma: Coupling constant
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :param plot_temporal_series_2: (boolean)
                                  Show temporal series (1) or not (0)
                                  of r1, R1, r2, R2, phi = phi1 - phi2,
                                  Phi = Phi1 - Phi2 when the weights are
                                  uniform in M
    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [[--- r ---]],
                               "r1",            [[--- r1---]],
                               "r2",            [[--- r2---]],
                               "r_uni",         [[--- r_uni ---]],
                               "r1_uni",        [[--- r1_uni ---]],
                               "r2_uni",        [[--- r2_uni ---]],
                               ...                ...
                               "R_none",        [[--- R_none ---]],
                               ...                ...
                               "R_all",         [[--- R_all ---]],
                               ...                ...
                               "R_hatredA",     [[--- R_hatredA ---]],
                               ...                ...
                               "R_hatLambda",   [[--- R_hatLambda ---]],
                               ...                ...
                               "R_uni",         [[--- R_uni ---]],
                               ...                ...
                               "p_out_array",    p_out_array,
                               "omega1_array",   omega1_array}
            where the values [[---X---]] is an array of shape
            len(p_out_list) times len(omega_1_list) of the order parameter X.
    """
    n1, n2 = sizes
    N = sum(sizes)
    R_dictionary = {}
    r_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r1_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r2_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    r_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r1_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    r2_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_none_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_none_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_none_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_all_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_all_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_all_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_hatredA_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_hatredA_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_hatredA_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_hatLambda_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_hatLambda_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_hatLambda_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    R_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R1_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))
    R2_uni_matrix = np.zeros((len(p_out_array), len(omega1_array)))

    for i in tqdm(range(len(p_out_array))):
        time.sleep(1)

        pq = [[0, p_out_array[i]], [p_out_array[i], 0]]
        P = (np.block([[np.ones(n1), np.zeros(n2)],
                       [np.zeros(n1), np.ones(n2)]])).T
        A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
        V = get_eigenvectors_matrix(A, 2)  # Not normalized
        Vp = pinv(V)
        # M_T = np.block([[V[0, 0:n1]/np.sum(V[0, 0:n1]), np.zeros(n2)],
        #                 [np.zeros(n1), V[0, n1:]/np.sum(V[0, n1:])]])

        M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                        [np.zeros(n1), 1 / n2 * np.ones(n2)]])
        C = np.dot(M_0, Vp)
        CV = np.dot(C, V)
        M = (CV.T / (np.sum(CV, axis=1))).T

        for j in range(len(omega1_array)):

            omega1 = omega1_array[j]
            omega2 = -n1/n2*omega1
            omega = np.array(n1 * [omega1] + n2 * [omega2])
            omega_array = np.array([omega1, omega2])

            # Integrate complete dynamics

            theta0 = 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            args_cosinus = (omega, sigma)
            cosinus_sol = integrate_dynamics(t0, t1, dt, cosinus, A,
                                             "dop853", theta0,
                                             *args_cosinus)

            r1 = np.absolute(
                np.sum(M[0, 0:n1] * np.exp(1j * cosinus_sol[:, 0:n1]),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, n1:] * np.exp(1j * cosinus_sol[:, n1:]),
                       axis=1))
            r = np.absolute(
                np.sum((n1*M[0, :] + n2*M[1, :])*np.exp(1j * cosinus_sol),
                       axis=1)) / N
            r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
            r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

            r1_uni = np.absolute(
                np.sum(M_0[0, 0:n1] * np.exp(1j * cosinus_sol[:, 0:n1]),
                       axis=1))
            r2_uni = np.absolute(
                np.sum(M_0[1, n1:] * np.exp(1j * cosinus_sol[:, n1:]),
                       axis=1))
            r_uni = np.absolute(
                np.sum((n1 * M_0[0, :] + n2 * M_0[1, :]) *
                       np.exp(1j * cosinus_sol), axis=1)) / N
            r1_uni_mean = np.sum(r1_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_uni_mean = np.sum(r2_uni[5 * int(t1 // dt) // 10:]
                                 ) / len(r2[5 * int(t1 // dt) // 10:])
            r_uni_mean = np.sum(r_uni[5 * int(t1 // dt) // 10:]
                                ) / len(r[5 * int(t1 // dt) // 10:])

            # Integrate recuced dynamics

            Z0 = np.dot(M, z0)
            redA = multi_dot([M, A, P])
            hatredA = (multi_dot([M**2, A, P]).T / np.diag(np.dot(M, M.T))).T
            hatLambda = multi_dot([M, A, pinv(M)])

            #  perturbation == "None":
            args_red_cosinus_2D = (omega_array, sigma, N,
                                   redA, redA)
            red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_cosinus_2D,
                                                 redA, "zvode", Z0,
                                                 *args_red_cosinus_2D)
            Z1, Z2 = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)

            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_none_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                  ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_none_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                  ) / len(R2[5 * int(t1 // dt) // 10:])
            R_none_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                 ) / len(R[5 * int(t1 // dt) // 10:])

            # perturbation == "All":
            args_red_cosinus_2D = (omega_array, sigma, N,
                                   hatredA, hatLambda)
            red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_cosinus_2D,
                                                 redA, "zvode", Z0,
                                                 *args_red_cosinus_2D)
            Z1, Z2 = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_all_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_all_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 ) / len(R2[5 * int(t1 // dt) // 10:])
            R_all_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                ) / len(R[5 * int(t1 // dt) // 10:])

            # perturbation == "hatredA":
            args_red_cosinus_2D = (omega_array, sigma, N,
                                   hatredA, redA)
            red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_cosinus_2D,
                                                 redA, "zvode", Z0,
                                                 *args_red_cosinus_2D)
            Z1, Z2 = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_hatredA_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                     ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_hatredA_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                     ) / len(R2[5 * int(t1 // dt) // 10:])
            R_hatredA_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                    ) / len(R[5 * int(t1 // dt) // 10:])

            # perturbation == "hatLambda":
            args_red_cosinus_2D = (omega_array, sigma, N,
                                   redA, hatLambda)
            red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_cosinus_2D,
                                                 redA, "zvode", Z0,
                                                 *args_red_cosinus_2D)
            Z1, Z2 = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_hatLambda_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                       ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_hatLambda_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                       ) / len(R2[5 * int(t1 // dt) // 10:])
            R_hatLambda_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                      ) / len(R[5 * int(t1 // dt) // 10:])

            # Uniform, perturbation == "None" and M = M_0 = M_T:
            Z0 = np.dot(M_0, z0)
            redA0 = multi_dot([M_0, A, P])
            args_red_cosinus_2D = (omega_array, sigma, N,
                                   redA0, redA0)
            red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_cosinus_2D,
                                                 redA0, "zvode", Z0,
                                                 *args_red_cosinus_2D)

            Z1, Z2 = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N
            R1_uni_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_uni_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 ) / len(R2[5 * int(t1 // dt) // 10:])
            R_uni_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                ) / len(R[5 * int(t1 // dt) // 10:])

            r_matrix[i, j] = r_mean
            r1_matrix[i, j] = r1_mean
            r2_matrix[i, j] = r2_mean

            r_uni_matrix[i, j] = r_uni_mean
            r1_uni_matrix[i, j] = r1_uni_mean
            r2_uni_matrix[i, j] = r2_uni_mean

            R_none_matrix[i, j] = R_none_mean
            R1_none_matrix[i, j] = R1_none_mean
            R2_none_matrix[i, j] = R2_none_mean

            R_all_matrix[i, j] = R_all_mean
            R1_all_matrix[i, j] = R1_all_mean
            R2_all_matrix[i, j] = R2_all_mean

            R_hatredA_matrix[i, j] = R_hatredA_mean
            R1_hatredA_matrix[i, j] = R1_hatredA_mean
            R2_hatredA_matrix[i, j] = R2_hatredA_mean

            R_hatLambda_matrix[i, j] = R_hatLambda_mean
            R1_hatLambda_matrix[i, j] = R1_hatLambda_mean
            R2_hatLambda_matrix[i, j] = R2_hatLambda_mean

            R_uni_matrix[i, j] = R_uni_mean
            R1_uni_matrix[i, j] = R1_uni_mean
            R2_uni_matrix[i, j] = R2_uni_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(6, 7))

                plt.subplot(311)
                plt.suptitle("$p = {}, \\omega_1 = {}, \\omega_2 = {}$"
                             .format(np.round(p_out_array[i], 3),
                                     np.round(omega1_array[j], 3),
                                     np.round(-n1/n2*omega1, 3)), y=1.0)
                plt.plot(r_uni, color="k")
                plt.plot(R, color="g")
                # plt.plot(r_mean * np.ones(len(r)), color="r")
                plt.ylabel("$R_{uni}$", fontsize=12)
                plt.ylim([-0.02, 1.02])

                plt.subplot(312)
                plt.plot(r1_uni, color="b")
                plt.plot(R1, color="lightblue")
                plt.ylabel("$(R_1)_{uni}$", fontsize=12)
                plt.ylim([-0.02, 1.02])

                plt.subplot(313)
                plt.plot(r2_uni, color="r")
                plt.plot(R2, color="y")
                plt.ylabel("$(R_2)_{uni}$", fontsize=12)
                plt.ylim([-0.02, 1.02])
                plt.xlabel("$t$", fontsize=12)

                plt.tight_layout()

                plt.show()

            if plot_temporal_series_2:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 10))

                Phi = np.angle(Z1) - np.angle(Z2)

                phi1_uni = np.angle(
                    np.sum(M_0[0, 0:n1] * np.exp(1j * cosinus_sol[:, 0:n1]),
                           axis=1))
                phi2_uni = np.angle(
                    np.sum(M_0[1, n1:] * np.exp(1j * cosinus_sol[:, n1:]),
                           axis=1))
                plt.subplot(311)
                plt.plot(r1_uni, color="k")
                plt.plot(R1, color="b", linestyle='-')
                plt.ylabel("$R_1$")

                plt.subplot(312)
                plt.plot(r2_uni, color="k")
                plt.plot(R2, color="r", linestyle='-')
                plt.ylabel("$R_2$")

                plt.subplot(313)
                # plt.scatter(t0, Phi[0], color="k", s=50)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1),
                            phi1_uni - phi2_uni,
                            color="k", s=10)
                plt.scatter(np.linspace(t0, t1, t1 // dt + 1), Phi,
                            color="purple", s=5)
                plt.ylabel("$\\Phi = \\Phi_1 - \\Phi_2$")
                plt.xlabel("Time $t$")
                plt.show()

    R_dictionary["r"] = r_matrix.tolist()
    R_dictionary["r1"] = r1_matrix.tolist()
    R_dictionary["r2"] = r2_matrix.tolist()

    R_dictionary["r_uni"] = r_uni_matrix.tolist()
    R_dictionary["r1_uni"] = r1_uni_matrix.tolist()
    R_dictionary["r2_uni"] = r2_uni_matrix.tolist()

    R_dictionary["R_none"] = R_none_matrix.tolist()
    R_dictionary["R1_none"] = R1_none_matrix.tolist()
    R_dictionary["R2_none"] = R2_none_matrix.tolist()

    R_dictionary["R_all"] = R_all_matrix.tolist()
    R_dictionary["R1_all"] = R1_all_matrix.tolist()
    R_dictionary["R2_all"] = R2_all_matrix.tolist()

    R_dictionary["R_hatredA"] = R_hatredA_matrix.tolist()
    R_dictionary["R1_hatredA"] = R1_hatredA_matrix.tolist()
    R_dictionary["R2_hatredA"] = R2_hatredA_matrix.tolist()

    R_dictionary["R_hatLambda"] = R_hatLambda_matrix.tolist()
    R_dictionary["R1_hatLambda"] = R1_hatLambda_matrix.tolist()
    R_dictionary["R2_hatLambda"] = R2_hatLambda_matrix.tolist()

    R_dictionary["R_uni"] = R_uni_matrix.tolist()
    R_dictionary["R1_uni"] = R1_uni_matrix.tolist()
    R_dictionary["R2_uni"] = R2_uni_matrix.tolist()

    R_dictionary["p_out_array"] = p_out_array.tolist()
    R_dictionary["omega1_array"] = omega1_array.tolist()

    return R_dictionary


def get_uniform_data_cosinus_2D_2(omega1_array, t0, t1, dt,
                                  sigma, plot_temporal_series=0,
                                  plot_temporal_series_2=0):
    """
    Generate data for different reduced cosinus dynamics on a small graph (two
    triangles connected by one of their nodes) when one module has one natural
    frequency and the other has another natural
    frequency (omega2 = -n1/n2*omega1).

    :param omega1_array: Array of the natural frequencies of the first
                         community
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param sigma: Coupling constant
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)
                                 of r, R, r1, R1, r2, R2 when the weights are
                                 uniform in M
    :param plot_temporal_series_2: (boolean)
                                  Show temporal series (1) or not (0)
                                  of r1, R1, r2, R2, phi = phi1 - phi2,
                                  Phi = Phi1 - Phi2 when the weights are
                                  uniform in M
    :return: R_dictionary
            R_dictionary is a dictionary of the form
                               Keys             Values
                             { "r",             [[--- r ---]],
                               "r1",            [[--- r1---]],
                               "r2",            [[--- r2---]],
                               "r_uni",         [[--- r_uni ---]],
                               "r1_uni",        [[--- r1_uni ---]],
                               "r2_uni",        [[--- r2_uni ---]],
                               ...                ...
                               "R_none",        [[--- R_none ---]],
                               ...                ...
                               "R_all",         [[--- R_all ---]],
                               ...                ...
                               "R_hatredA",     [[--- R_hatredA ---]],
                               ...                ...
                               "R_hatLambda",   [[--- R_hatLambda ---]],
                               ...                ...
                               "R_uni",         [[--- R_uni ---]],
                               ...                ...
                               "omega1_array",   omega1_array}
            where the values [[---X---]] is an array of shape
            1 times len(omega_1_list) of the order parameter X.
    """
    R_dictionary = {}
    r_matrix = np.zeros(len(omega1_array))
    r1_matrix = np.zeros(len(omega1_array))
    r2_matrix = np.zeros(len(omega1_array))

    r_uni_matrix = np.zeros(len(omega1_array))
    r1_uni_matrix = np.zeros(len(omega1_array))
    r2_uni_matrix = np.zeros(len(omega1_array))

    R_none_matrix = np.zeros(len(omega1_array))
    R1_none_matrix = np.zeros(len(omega1_array))
    R2_none_matrix = np.zeros(len(omega1_array))

    R_all_matrix = np.zeros(len(omega1_array))
    R1_all_matrix = np.zeros(len(omega1_array))
    R2_all_matrix = np.zeros(len(omega1_array))

    R_hatredA_matrix = np.zeros(len(omega1_array))
    R1_hatredA_matrix = np.zeros(len(omega1_array))
    R2_hatredA_matrix = np.zeros(len(omega1_array))

    R_hatLambda_matrix = np.zeros(len(omega1_array))
    R1_hatLambda_matrix = np.zeros(len(omega1_array))
    R2_hatLambda_matrix = np.zeros(len(omega1_array))

    R_uni_matrix = np.zeros(len(omega1_array))
    R1_uni_matrix = np.zeros(len(omega1_array))
    R2_uni_matrix = np.zeros(len(omega1_array))

    n1, n2 = 3, 3
    N = 6
    A = np.array([[0, 1, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [1, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 1],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1, 0]])
    P = (np.block([[np.ones(n1), np.zeros(n2)],
                   [np.zeros(n1), np.ones(n2)]])).T
    # V = get_eigenvectors_matrix(A, 2)  # Not normalized
    # Vp = pinv(V)
    # M_T = np.block([[V[0, 0:n1]/np.sum(V[0, 0:n1]), np.zeros(n2)],
    #                 [np.zeros(n1), V[0, n1:]/np.sum(V[0, n1:])]])

    M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                    [np.zeros(n1), 1 / n2 * np.ones(n2)]])
    # C = np.dot(M_0, Vp)
    # CV = np.dot(C, V)
    M = np.array([[0.2929, 0.2929, 0.3143, 0.0999, 0, 0],
                  [0, 0, 0.0999, 0.3143, 0.2929, 0.2929]])
    # (CV.T / (np.sum(CV, axis=1))).T
    for j in tqdm(range(len(omega1_array))):
        time.sleep(1)

        omega1 = omega1_array[j]
        omega2 = -n1/n2*omega1
        omega = np.array(n1 * [omega1] + n2 * [omega2])
        omega_array = np.array([omega1, omega2])

        # Integrate complete dynamics

        theta0 = 2 * np.pi * np.random.rand(N)
        z0 = np.exp(1j * theta0)
        args_cosinus = (omega, sigma)
        cosinus_sol = integrate_dynamics(t0, t1, dt, cosinus, A,
                                         "dop853", theta0,
                                         *args_cosinus)

        r1 = np.absolute(
            np.sum(M[0, 0:n1] * np.exp(1j * cosinus_sol[:, 0:n1]),
                   axis=1))
        r2 = np.absolute(
            np.sum(M[1, n1:] * np.exp(1j * cosinus_sol[:, n1:]),
                   axis=1))
        r = np.absolute(
            np.sum((n1*M[0, :] + n2*M[1, :])*np.exp(1j * cosinus_sol),
                   axis=1)) / N
        r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                         ) / len(r1[5 * int(t1 // dt) // 10:])
        r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                         ) / len(r2[5 * int(t1 // dt) // 10:])
        r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                        ) / len(r[5 * int(t1 // dt) // 10:])

        r1_uni = np.absolute(
            np.sum(M_0[0, 0:n1] * np.exp(1j * cosinus_sol[:, 0:n1]),
                   axis=1))
        r2_uni = np.absolute(
            np.sum(M_0[1, n1:] * np.exp(1j * cosinus_sol[:, n1:]),
                   axis=1))
        r_uni = np.absolute(
            np.sum((n1 * M_0[0, :] + n2 * M_0[1, :]) *
                   np.exp(1j * cosinus_sol), axis=1)) / N
        r1_uni_mean = np.sum(r1_uni[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
        r2_uni_mean = np.sum(r2_uni[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
        r_uni_mean = np.sum(r_uni[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

        # Integrate recuced dynamics

        Z0 = np.dot(M, z0)
        redA = multi_dot([M, A, P])
        hatredA = (multi_dot([M**2, A, P]).T / np.diag(np.dot(M, M.T))).T
        hatLambda = multi_dot([M, A, pinv(M)])

        #  perturbation == "None":
        args_red_cosinus_2D = (omega_array, sigma, N,
                               redA, redA)
        red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                             reduced_cosinus_2D,
                                             redA, "zvode", Z0,
                                             *args_red_cosinus_2D)
        Z1_none, Z2_none = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
        R1_none, R2_none = np.absolute(Z1_none), np.absolute(Z2_none)

        R_none = np.absolute(n1 * Z1_none + n2 * Z2_none) / N
        R1_none_mean = np.sum(R1_none[5 * int(t1 // dt) // 10:]
                              ) / len(R1_none[5 * int(t1 // dt) // 10:])
        R2_none_mean = np.sum(R2_none[5 * int(t1 // dt) // 10:]
                              ) / len(R2_none[5 * int(t1 // dt) // 10:])
        R_none_mean = np.sum(R_none[5 * int(t1 // dt) // 10:]
                             ) / len(R_none[5 * int(t1 // dt) // 10:])

        # perturbation == "All":
        args_red_cosinus_2D = (omega_array, sigma, N,
                               hatredA, hatLambda)
        red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                             reduced_cosinus_2D,
                                             redA, "zvode", Z0,
                                             *args_red_cosinus_2D)
        Z1, Z2 = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_all_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                             ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_all_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                             ) / len(R2[5 * int(t1 // dt) // 10:])
        R_all_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                            ) / len(R[5 * int(t1 // dt) // 10:])

        # perturbation == "hatredA":
        args_red_cosinus_2D = (omega_array, sigma, N,
                               hatredA, redA)
        red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                             reduced_cosinus_2D,
                                             redA, "zvode", Z0,
                                             *args_red_cosinus_2D)
        Z1, Z2 = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_hatredA_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_hatredA_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 ) / len(R2[5 * int(t1 // dt) // 10:])
        R_hatredA_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                ) / len(R[5 * int(t1 // dt) // 10:])

        # perturbation == "hatLambda":
        args_red_cosinus_2D = (omega_array, sigma, N,
                               redA, hatLambda)
        red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                             reduced_cosinus_2D,
                                             redA, "zvode", Z0,
                                             *args_red_cosinus_2D)
        Z1, Z2 = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_hatLambda_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                                   ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_hatLambda_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                                   ) / len(R2[5 * int(t1 // dt) // 10:])
        R_hatLambda_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                                  ) / len(R[5 * int(t1 // dt) // 10:])

        # Uniform, perturbation == "None" and M = M_0 = M_T:
        Z0 = np.dot(M_0, z0)
        redA0 = multi_dot([M_0, A, P])
        args_red_cosinus_2D = (omega_array, sigma, N,
                               redA0, redA0)
        red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                             reduced_cosinus_2D,
                                             redA0, "zvode", Z0,
                                             *args_red_cosinus_2D)

        Z1, Z2 = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_uni_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                             ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_uni_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                             ) / len(R2[5 * int(t1 // dt) // 10:])
        R_uni_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                            ) / len(R[5 * int(t1 // dt) // 10:])

        r_matrix[j] = r_mean
        r1_matrix[j] = r1_mean
        r2_matrix[j] = r2_mean

        r_uni_matrix[j] = r_uni_mean
        r1_uni_matrix[j] = r1_uni_mean
        r2_uni_matrix[j] = r2_uni_mean

        R_none_matrix[j] = R_none_mean
        R1_none_matrix[j] = R1_none_mean
        R2_none_matrix[j] = R2_none_mean

        R_all_matrix[j] = R_all_mean
        R1_all_matrix[j] = R1_all_mean
        R2_all_matrix[j] = R2_all_mean

        R_hatredA_matrix[j] = R_hatredA_mean
        R1_hatredA_matrix[j] = R1_hatredA_mean
        R2_hatredA_matrix[j] = R2_hatredA_mean

        R_hatLambda_matrix[j] = R_hatLambda_mean
        R1_hatLambda_matrix[j] = R1_hatLambda_mean
        R2_hatLambda_matrix[j] = R2_hatLambda_mean

        R_uni_matrix[j] = R_uni_mean
        R1_uni_matrix[j] = R1_uni_mean
        R2_uni_matrix[j] = R2_uni_mean

        if plot_temporal_series:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 7))

            plt.subplot(311)
            plt.suptitle("\\omega_1 = {}, \\omega_2 = {}$"
                         .format(np.round(omega1_array[j], 3),
                                 np.round(-n1/n2*omega1, 3)), y=1.0)
            plt.plot(r_uni, color="k")
            plt.plot(R, color="g")
            plt.plot(R_none, color="r")
            # plt.plot(r_mean * np.ones(len(r)), color="r")
            plt.ylabel("$R_{uni}$", fontsize=12)
            plt.ylim([-0.02, 1.02])

            plt.subplot(312)
            plt.plot(r1_uni, color="b")
            plt.plot(R1, color="lightblue")
            plt.plot(R1_none, color="purple")
            plt.ylabel("$(R_1)_{uni}$", fontsize=12)
            plt.ylim([-0.02, 1.02])

            plt.subplot(313)
            plt.plot(r2_uni, color="r")
            plt.plot(R2, color="y")
            plt.plot(R2_none, color="orange")
            plt.ylabel("$(R_2)_{uni}$", fontsize=12)
            plt.ylim([-0.02, 1.02])
            plt.xlabel("$t$", fontsize=12)

            plt.tight_layout()

            plt.show()

        if plot_temporal_series_2:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10))

            Phi = np.angle(Z1) - np.angle(Z2)

            phi1_uni = np.angle(
                np.sum(M_0[0, 0:n1] * np.exp(1j * cosinus_sol[:, 0:n1]),
                       axis=1))
            phi2_uni = np.angle(
                np.sum(M_0[1, n1:] * np.exp(1j * cosinus_sol[:, n1:]),
                       axis=1))
            plt.subplot(311)
            plt.plot(r1_uni, color="k")
            plt.plot(R1, color="b", linestyle='-')
            plt.ylabel("$R_1$")

            plt.subplot(312)
            plt.plot(r2_uni, color="k")
            plt.plot(R2, color="r", linestyle='-')
            plt.ylabel("$R_2$")

            plt.subplot(313)
            # plt.scatter(t0, Phi[0], color="k", s=50)
            plt.scatter(np.linspace(t0, t1, t1 // dt + 1),
                        phi1_uni - phi2_uni,
                        color="k", s=10)
            plt.scatter(np.linspace(t0, t1, t1 // dt + 1), Phi,
                        color="purple", s=5)
            plt.ylabel("$\\Phi = \\Phi_1 - \\Phi_2$")
            plt.xlabel("Time $t$")
            plt.show()

    R_dictionary["r"] = r_matrix.tolist()
    R_dictionary["r1"] = r1_matrix.tolist()
    R_dictionary["r2"] = r2_matrix.tolist()

    R_dictionary["r_uni"] = r_uni_matrix.tolist()
    R_dictionary["r1_uni"] = r1_uni_matrix.tolist()
    R_dictionary["r2_uni"] = r2_uni_matrix.tolist()

    R_dictionary["R_none"] = R_none_matrix.tolist()
    R_dictionary["R1_none"] = R1_none_matrix.tolist()
    R_dictionary["R2_none"] = R2_none_matrix.tolist()

    R_dictionary["R_all"] = R_all_matrix.tolist()
    R_dictionary["R1_all"] = R1_all_matrix.tolist()
    R_dictionary["R2_all"] = R2_all_matrix.tolist()

    R_dictionary["R_hatredA"] = R_hatredA_matrix.tolist()
    R_dictionary["R1_hatredA"] = R1_hatredA_matrix.tolist()
    R_dictionary["R2_hatredA"] = R2_hatredA_matrix.tolist()

    R_dictionary["R_hatLambda"] = R_hatLambda_matrix.tolist()
    R_dictionary["R1_hatLambda"] = R1_hatLambda_matrix.tolist()
    R_dictionary["R2_hatLambda"] = R2_hatLambda_matrix.tolist()

    R_dictionary["R_uni"] = R_uni_matrix.tolist()
    R_dictionary["R1_uni"] = R1_uni_matrix.tolist()
    R_dictionary["R2_uni"] = R2_uni_matrix.tolist()

    R_dictionary["omega1_array"] = omega1_array.tolist()

    return R_dictionary


def get_random_data_winfree_2D(sizes, nb_data, omega_min, omega_max,
                               t0, t1, dt, sigma, plot_temporal_series):
    """
    Prediction errors for the Winfree dynamics on SBM graphs
    when one module has one natural frequency and the other has
    another natural frequency

    :param sizes: [n1, n2] Community sizes
    :param nb_data: The number of instances of graphs, initial conditions
    :param omega_min: Minimum value of omega
    :param omega_max: Maximum value of omega
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param sigma: Coupling constant
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)

    :return:
            R_matrix is a 21 by nb_data matrix of the form
                        [[--- r ---],
                         [--- r_uni ---],
                         [--- R_none ---],
                         [--- R_all ---],
                         [--- R_hatredA ---],
                         [--- R_hatLambda ---],
                         [--- R_uni ---]]
            parameters_matrix is a 2 by nb_data matrix of the form
                         [[--- p_out ---],
                         [--- omega1 ---],
                         [--- omega_2 ---]]"

    """
    n1, n2 = sizes
    N = sum(sizes)
    R_matrix = np.zeros((21, nb_data))
    parameters_matrix = np.zeros((3, nb_data))

    for data in tqdm(range(nb_data)):

        time.sleep(2)

        p_out = (1 - 1/np.sqrt(N)) * np.random.random() + 1/np.sqrt(N)
        omega1 = (omega_max - omega_min) * np.random.random() + omega_min
        omega2 = (omega_max - omega_min) * np.random.random() + omega_min
        omega = np.array(n1 * [omega1] + n2 * [omega2])
        omega_array = np.array([omega1, omega2])

        parameters_matrix[0, data] = p_out
        parameters_matrix[1, data] = omega1
        parameters_matrix[2, data] = omega2

        pq = [[0, p_out], [p_out, 0]]
        M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                        [np.zeros(n1), 1 / n2 * np.ones(n2)]])
        P = (np.block([[np.ones(n1), np.zeros(n2)],
                       [np.zeros(n1), np.ones(n2)]])).T
        A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
        V = get_eigenvectors_matrix(A, 2)  # Not normalized
        Vp = pinv(V)
        C = np.dot(M_0, Vp)
        CV = np.dot(C, V)
        M = (CV.T / (np.sum(CV, axis=1))).T

        # Integrate complete dynamics

        theta0 = 2 * np.pi * np.random.rand(N)
        z0 = np.exp(1j * theta0)
        args_winfree = (omega, sigma)
        winfree_sol = integrate_dynamics(t0, t1, dt, winfree, A,
                                         "dop853", theta0,
                                         *args_winfree)

        r1 = np.absolute(
            np.sum(M[0, 0:n1] * np.exp(1j * winfree_sol[:, 0:n1]),
                   axis=1))
        r2 = np.absolute(
            np.sum(M[1, n1:] * np.exp(1j * winfree_sol[:, n1:]),
                   axis=1))
        r = np.absolute(
            np.sum((n1 * M[0, :] + n2 * M[1, :]) * np.exp(1j * winfree_sol),
                   axis=1)) / N

        r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                         ) / len(r1[5 * int(t1 // dt) // 10:])
        r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                         ) / len(r2[5 * int(t1 // dt) // 10:])
        r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                        ) / len(r[5 * int(t1 // dt) // 10:])

        r1_uni = np.absolute(
            np.sum(M_0[0, 0:n1] * np.exp(1j * winfree_sol[:, 0:n1]),
                   axis=1))
        r2_uni = np.absolute(
            np.sum(M_0[1, n1:] * np.exp(1j * winfree_sol[:, n1:]),
                   axis=1))
        r_uni = np.absolute(
            np.sum((n1 * M_0[0, :] + n2 * M_0[1, :]) *
                   np.exp(1j * winfree_sol), axis=1)) / N

        r1_uni_mean = np.sum(r1_uni[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
        r2_uni_mean = np.sum(r2_uni[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
        r_uni_mean = np.sum(r_uni[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

        # Integrate recuced dynamics

        Z0 = np.dot(M, z0)
        redA = multi_dot([M, A, P])
        hatredA = (multi_dot([M ** 2, A, P]).T / np.diag(np.dot(M, M.T))).T
        hatLambda = multi_dot([M, A, pinv(M)])

        #  perturbation == "None":
        args_red_winfree_2D = (omega_array, sigma, N,
                               redA, redA)
        red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                             reduced_winfree_2D,
                                             redA, "zvode", Z0,
                                             *args_red_winfree_2D)
        Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_mean_none = np.sum(R1[5 * int(t1 // dt) // 10:]
                              ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_mean_none = np.sum(R2[5 * int(t1 // dt) // 10:]
                              ) / len(R2[5 * int(t1 // dt) // 10:])
        R_mean_none = np.sum(R[5 * int(t1 // dt) // 10:]
                             ) / len(R[5 * int(t1 // dt) // 10:])

        # perturbation == "All":
        args_red_winfree_2D = (omega_array, sigma, N,
                               hatredA, hatLambda)
        red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                             reduced_winfree_2D,
                                             redA, "zvode", Z0,
                                             *args_red_winfree_2D)
        Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_mean_all = np.sum(R1[5 * int(t1 // dt) // 10:]
                             ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_mean_all = np.sum(R2[5 * int(t1 // dt) // 10:]
                             ) / len(R2[5 * int(t1 // dt) // 10:])
        R_mean_all = np.sum(R[5 * int(t1 // dt) // 10:]
                            ) / len(R[5 * int(t1 // dt) // 10:])

        # perturbation == "hatredA":
        args_red_winfree_2D = (omega_array, sigma, N,
                               hatredA, redA)
        red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                             reduced_winfree_2D,
                                             redA, "zvode", Z0,
                                             *args_red_winfree_2D)
        Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_mean_hatredA = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_mean_hatredA = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 ) / len(R2[5 * int(t1 // dt) // 10:])
        R_mean_hatredA = np.sum(R[5 * int(t1 // dt) // 10:]
                                ) / len(R[5 * int(t1 // dt) // 10:])

        # perturbation == "hatLambda":
        args_red_winfree_2D = (omega_array, sigma, N,
                               redA, hatLambda)
        red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                             reduced_winfree_2D,
                                             redA, "zvode", Z0,
                                             *args_red_winfree_2D)
        Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_mean_hatLambda = np.sum(R1[5 * int(t1 // dt) // 10:]
                                   ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_mean_hatLambda = np.sum(R2[5 * int(t1 // dt) // 10:]
                                   ) / len(R2[5 * int(t1 // dt) // 10:])
        R_mean_hatLambda = np.sum(R[5 * int(t1 // dt) // 10:]
                                  ) / len(R[5 * int(t1 // dt) // 10:])

        # Uniform, perturbation == "None" and M = M_0 = M_T:
        Z0 = np.dot(M_0, z0)
        redA0 = multi_dot([M_0, A, P])
        args_red_winfree_2D = (omega_array, sigma, N,
                               redA0, redA0)
        red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                             reduced_winfree_2D,
                                             redA, "zvode", Z0,
                                             *args_red_winfree_2D)

        Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_mean_uni = np.sum(R1[5 * int(t1 // dt) // 10:]
                             ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_mean_uni = np.sum(R2[5 * int(t1 // dt) // 10:]
                             ) / len(R2[5 * int(t1 // dt) // 10:])
        R_mean_uni = np.sum(R[5 * int(t1 // dt) // 10:]
                            ) / len(R[5 * int(t1 // dt) // 10:])

        R_matrix[0, data] = r_mean
        R_matrix[1, data] = r1_mean
        R_matrix[2, data] = r2_mean
        R_matrix[3, data] = r_uni_mean
        R_matrix[4, data] = r1_uni_mean
        R_matrix[5, data] = r2_uni_mean
        perturbations = ["none", "all", "hatredA", "hatLambda", "uni"]
        R_mean_list = [R_mean_none, R_mean_all, R_mean_hatredA,
                       R_mean_hatLambda, R_mean_uni]
        R1_mean_list = [R1_mean_none, R1_mean_all, R1_mean_hatredA,
                        R1_mean_hatLambda, R1_mean_uni]
        R2_mean_list = [R2_mean_none, R2_mean_all, R2_mean_hatredA,
                        R2_mean_hatLambda, R2_mean_uni]
        j = 6
        for i in range(len(perturbations)):
            R_matrix[j, data] = R_mean_list[i]
            R_matrix[j+1, data] = R1_mean_list[i]
            R_matrix[j+2, data] = R2_mean_list[i]
            j += 3

        if plot_temporal_series:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 7))

            plt.subplot(311)
            plt.suptitle("$p = {}, \\omega_1 = {}, \\omega_2 = {}$"
                         .format(np.round(p_out, 3), np.round(omega1, 3),
                                 np.round(omega2, 3)), y=1.0)
            plt.plot(r_uni, color="k")
            plt.plot(R, color="g")
            # plt.plot(r_mean * np.ones(len(r)), color="r")
            plt.ylabel("$R$", fontsize=12)
            plt.ylim([-0.02, 1.02])

            plt.subplot(312)
            plt.plot(r1_uni, color="b")
            plt.plot(R1, color="lightblue")
            plt.ylabel("$R_1$", fontsize=12)
            plt.ylim([-0.02, 1.02])

            plt.subplot(313)
            plt.plot(r2_uni, color="r")
            plt.plot(R2, color="y")
            plt.ylabel("$R_2$", fontsize=12)
            plt.ylim([-0.02, 1.02])
            plt.xlabel("$t$", fontsize=12)

            plt.tight_layout()

            plt.show()

    return R_matrix, parameters_matrix


def get_random_data_kuramoto_2D(sizes, nb_data, omega_min, omega_max,
                                t0, t1, dt, sigma, plot_temporal_series):
    """
    Prediction errors for the Kuramoto dynamics on SBM graphs
    when one module has one natural frequency and the other has
    another natural frequency

    :param sizes: [n1, n2] Community sizes
    :param nb_data: The number of instances of graphs, initial conditions
    :param omega_min: Minimum value of omega
    :param omega_max: Maximum value of omega
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param sigma: Coupling constant
    :param plot_temporal_series: (boolean) Show temporal series (1) or not (0)

    :return:
            R_matrix is a 21 by nb_data matrix of the form
                        [[--- r ---],
                         [--- r1 ---],
                         [--- r2 ---],
                         ...
                         [--- r_uni ---],
                         ...
                         [--- R_none ---],
                         ...
                         [--- R_all ---],
                         ...
                         [--- R_hatredA ---],
                         ...
                         [--- R_hatLambda ---],
                         ...
                         [--- R_uni ---]]
            parameters_matrix is a 2 by nb_data matrix of the form
                         [[--- p_out ---],
                         [--- omega1 ---],
                         [--- omega_2 ---]]"

    """
    n1, n2 = sizes
    N = sum(sizes)
    R_matrix = np.zeros((21, nb_data))
    parameters_matrix = np.zeros((3, nb_data))

    for data in tqdm(range(nb_data)):

        time.sleep(2)

        p_out = (1 - 1/np.sqrt(N)) * np.random.random() + 1/np.sqrt(N)
        omega1 = (omega_max - omega_min) * np.random.random() + omega_min
        omega2 = (omega_max - omega_min) * np.random.random() + omega_min
        omega = np.array(n1 * [omega1] + n2 * [omega2])
        omega_array = np.array([omega1, omega2])

        parameters_matrix[0, data] = p_out
        parameters_matrix[1, data] = omega1
        parameters_matrix[2, data] = omega2

        pq = [[0, p_out], [p_out, 0]]
        M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                        [np.zeros(n1), 1 / n2 * np.ones(n2)]])
        P = (np.block([[np.ones(n1), np.zeros(n2)],
                       [np.zeros(n1), np.ones(n2)]])).T
        A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
        V = get_eigenvectors_matrix(A, 2)  # Not normalized
        Vp = pinv(V)
        C = np.dot(M_0, Vp)
        CV = np.dot(C, V)
        M = (CV.T / (np.sum(CV, axis=1))).T

        # Integrate complete dynamics

        theta0 = 2 * np.pi * np.random.rand(N)
        z0 = np.exp(1j * theta0)
        args_kuramoto = (omega, sigma)
        kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                          "dop853", theta0,
                                          *args_kuramoto)

        r1 = np.absolute(
            np.sum(M[0, 0:n1] * np.exp(1j * kuramoto_sol[:, 0:n1]),
                   axis=1))
        r2 = np.absolute(
            np.sum(M[1, n1:] * np.exp(1j * kuramoto_sol[:, n1:]),
                   axis=1))
        r = np.absolute(
            np.sum((n1 * M[0, :] + n2 * M[1, :]) * np.exp(1j * kuramoto_sol),
                   axis=1)) / N

        r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                         ) / len(r1[5 * int(t1 // dt) // 10:])
        r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                         ) / len(r2[5 * int(t1 // dt) // 10:])
        r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                        ) / len(r[5 * int(t1 // dt) // 10:])

        r1_uni = np.absolute(
            np.sum(M_0[0, 0:n1] * np.exp(1j * kuramoto_sol[:, 0:n1]),
                   axis=1))
        r2_uni = np.absolute(
            np.sum(M_0[1, n1:] * np.exp(1j * kuramoto_sol[:, n1:]),
                   axis=1))
        r_uni = np.absolute(
            np.sum((n1 * M_0[0, :] + n2 * M_0[1, :]) *
                   np.exp(1j * kuramoto_sol), axis=1)) / N

        r1_uni_mean = np.sum(r1_uni[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
        r2_uni_mean = np.sum(r2_uni[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
        r_uni_mean = np.sum(r_uni[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

        # Integrate recuced dynamics

        Z0 = np.dot(M, z0)
        redA = multi_dot([M, A, P])
        hatredA = (multi_dot([M ** 2, A, P]).T / np.diag(np.dot(M, M.T))).T
        hatLambda = multi_dot([M, A, pinv(M)])

        #  perturbation == "None":
        args_red_kuramoto_2D = (omega_array, sigma, N,
                                redA, redA)
        red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                              reduced_kuramoto_2D,
                                              redA, "zvode", Z0,
                                              *args_red_kuramoto_2D)
        Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_mean_none = np.sum(R1[5 * int(t1 // dt) // 10:]
                              ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_mean_none = np.sum(R2[5 * int(t1 // dt) // 10:]
                              ) / len(R2[5 * int(t1 // dt) // 10:])
        R_mean_none = np.sum(R[5 * int(t1 // dt) // 10:]
                             ) / len(R[5 * int(t1 // dt) // 10:])

        # perturbation == "All":
        args_red_kuramoto_2D = (omega_array, sigma, N,
                                hatredA, hatLambda)
        red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                              reduced_kuramoto_2D,
                                              redA, "zvode", Z0,
                                              *args_red_kuramoto_2D)
        Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_mean_all = np.sum(R1[5 * int(t1 // dt) // 10:]
                             ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_mean_all = np.sum(R2[5 * int(t1 // dt) // 10:]
                             ) / len(R2[5 * int(t1 // dt) // 10:])
        R_mean_all = np.sum(R[5 * int(t1 // dt) // 10:]
                            ) / len(R[5 * int(t1 // dt) // 10:])

        # perturbation == "hatredA":
        args_red_kuramoto_2D = (omega_array, sigma, N,
                                hatredA, redA)
        red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                              reduced_kuramoto_2D,
                                              redA, "zvode", Z0,
                                              *args_red_kuramoto_2D)
        Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_mean_hatredA = np.sum(R1[5 * int(t1 // dt) // 10:]
                                 ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_mean_hatredA = np.sum(R2[5 * int(t1 // dt) // 10:]
                                 ) / len(R2[5 * int(t1 // dt) // 10:])
        R_mean_hatredA = np.sum(R[5 * int(t1 // dt) // 10:]
                                ) / len(R[5 * int(t1 // dt) // 10:])

        # perturbation == "hatLambda":
        args_red_kuramoto_2D = (omega_array, sigma, N,
                                redA, hatLambda)
        red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                              reduced_kuramoto_2D,
                                              redA, "zvode", Z0,
                                              *args_red_kuramoto_2D)
        Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_mean_hatLambda = np.sum(R1[5 * int(t1 // dt) // 10:]
                                   ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_mean_hatLambda = np.sum(R2[5 * int(t1 // dt) // 10:]
                                   ) / len(R2[5 * int(t1 // dt) // 10:])
        R_mean_hatLambda = np.sum(R[5 * int(t1 // dt) // 10:]
                                  ) / len(R[5 * int(t1 // dt) // 10:])

        # Uniform, perturbation == "None" and M = M_0 = M_T:
        Z0 = np.dot(M_0, z0)
        redA0 = multi_dot([M_0, A, P])
        args_red_kuramoto_2D = (omega_array, sigma, N,
                                redA0, redA0)
        red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                              reduced_kuramoto_2D,
                                              redA, "zvode", Z0,
                                              *args_red_kuramoto_2D)

        Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
        R1, R2 = np.absolute(Z1), np.absolute(Z2)
        R = np.absolute(n1 * Z1 + n2 * Z2) / N
        R1_mean_uni = np.sum(R1[5 * int(t1 // dt) // 10:]
                             ) / len(R1[5 * int(t1 // dt) // 10:])
        R2_mean_uni = np.sum(R2[5 * int(t1 // dt) // 10:]
                             ) / len(R2[5 * int(t1 // dt) // 10:])
        R_mean_uni = np.sum(R[5 * int(t1 // dt) // 10:]
                            ) / len(R[5 * int(t1 // dt) // 10:])

        R_matrix[0, data] = r_mean
        R_matrix[1, data] = r1_mean
        R_matrix[2, data] = r2_mean
        R_matrix[3, data] = r_uni_mean
        R_matrix[4, data] = r1_uni_mean
        R_matrix[5, data] = r2_uni_mean
        perturbations = ["none", "all", "hatredA", "hatLambda", "uni"]
        R_mean_list = [R_mean_none, R_mean_all, R_mean_hatredA,
                       R_mean_hatLambda, R_mean_uni]
        R1_mean_list = [R1_mean_none, R1_mean_all, R1_mean_hatredA,
                        R1_mean_hatLambda, R1_mean_uni]
        R2_mean_list = [R2_mean_none, R2_mean_all, R2_mean_hatredA,
                        R2_mean_hatLambda, R2_mean_uni]
        j = 6
        for i in range(len(perturbations)):
            R_matrix[j, data] = R_mean_list[i]
            R_matrix[j+1, data] = R1_mean_list[i]
            R_matrix[j+2, data] = R2_mean_list[i]
            j += 3

        if plot_temporal_series:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 7))

            plt.subplot(311)
            plt.suptitle("$p = {}, \\omega_1 = {}, \\omega_2 = {}$"
                         .format(np.round(p_out, 3), np.round(omega1, 3),
                                 np.round(omega2, 3)), y=1.0)
            plt.plot(r_uni, color="k")
            plt.plot(R, color="g")
            # plt.plot(r_mean * np.ones(len(r)), color="r")
            plt.ylabel("$R$", fontsize=12)
            plt.ylim([-0.02, 1.02])

            plt.subplot(312)
            plt.plot(r1_uni, color="b")
            plt.plot(R1, color="lightblue")
            plt.ylabel("$R_1$", fontsize=12)
            plt.ylim([-0.02, 1.02])

            plt.subplot(313)
            plt.plot(r2_uni, color="r")
            plt.plot(R2, color="y")
            plt.ylabel("$R_2$", fontsize=12)
            plt.ylim([-0.02, 1.02])
            plt.xlabel("$t$", fontsize=12)

            plt.tight_layout()

            plt.show()

    return R_matrix, parameters_matrix


# def get_reduction_RMSE_winfree_2D_old(sizes, nb_data, omega_min, omega_max,
#                                       t0, t1, dt, sigma, N, SBM=False,
#                                       mean_SBM=False,
#                                       plot_temporal_series=False,
#                                       uniform=False, perturbation="None"):
#     """
#     Prediction errors for the Winfree dynamics on modular graphs
#     when one module has one natural frequency and the other has
#     another natural frequency
#
#     :param sizes: [n1, n2] Community sizes
#     :param nb_data: The number of instances of graphs, initial conditions
#     :param omega_min: Minimum value of omega
#     :param omega_max: Maximum value of omega
#     :param t0: Initial time
#     :param t1: Final time
#     :param dt: Time step
#     :param sigma: Coupling constant
#     :param N: Number of nodes
#     :param SBM: (Boolean)If we want to get the RMSE for the
#                  stochastic block model, it is True.
#     :param mean_SBM: (Boolean)If we want to get the RMSE for the mean
#                      stochastic block model, it is True.
#     :param plot_temporal_series: if True, the plot R, R1 and R2 vs time
#                                  is shown
#     :param uniform:
#     :param perturbation:
#
#     :return: RMSE for each community (RMSE1 and RMSE2) and the global RMSE
#
#     """
#     n1, n2 = sizes
#
#     RMSE1, RMSE2, RMSE_tot = 0, 0, 0
#
#     for _ in tqdm(range(nb_data)):
#
#         time.sleep(2)
#
#         p_out = np.random.random()
#
#         if SBM is True or mean_SBM is True:
#             p_in = np.random.random()
#         else:
#             p_in = 0
#
#         omega1 = (omega_max - omega_min)*np.random.random() + omega_min
#         omega2 = (omega_max - omega_min)*np.random.random() + omega_min
#         omega = np.array(n1*[omega1] + n2*[omega2])
#         omega_array = np.array([omega1, omega2])
#
#         pq = [[p_in, p_out], [p_out, p_in]]
#
#         M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
#                         [np.zeros(n1), 1 / n2 * np.ones(n2)]])
#         P = (np.block([[np.ones(n1), np.zeros(n2)],
#                        [np.zeros(n1), np.ones(n2)]])).T
#
#         if mean_SBM:
#             A = np.zeros((N, N))
#             ii = 0
#             for i in range(0, len(sizes)):
#                 jj = 0
#                 for j in range(0, len(sizes)):
#                     A[ii:ii + sizes[i], jj:jj + sizes[j]] \
#                         = pq[i][j] * np.ones((sizes[i], sizes[j]))
#                     jj += sizes[j]
#                 ii += sizes[i]
#
#             M = M_0
#
#         else:
#             A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
#             if p_in > 0:
#                 M = M_0
#             elif uniform:
#                 M = M_0
#             else:
#                 V = get_eigenvectors_matrix(A, 2)  # Not normalized
#                 Vp = pinv(V)
#                 C = np.dot(M_0, Vp)
#                 CV = np.dot(C, V)
#                 M = (CV.T / (np.sum(CV, axis=1))).T
#
#         theta0 = 2*np.pi*np.random.rand(N)
#         z0 = np.exp(1j * theta0)
#         Z0 = np.dot(M, z0)
#
#         # Integrate complete dynamics
#         args_winfree = (omega, sigma)
#         winfree_sol = integrate_dynamics(t0, t1, dt, winfree, A,
#                                          "dop853", theta0,
#                                          *args_winfree)
#
#         r1 = np.absolute(
#             np.sum(M[0, 0:n1] * np.exp(1j * winfree_sol[:, 0:n1]),
#                    axis=1))
#         r2 = np.absolute(
#             np.sum(M[1, n1:] * np.exp(1j * winfree_sol[:, n1:]),
#                    axis=1))
#         r = np.absolute(
#             np.sum((n1*M[0, :] + n2*M[1, :]) * np.exp(1j * winfree_sol),
#                    axis=1))/N
#
#         r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
#                          ) / len(r1[5 * int(t1 // dt) // 10:])
#         r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
#                          ) / len(r2[5 * int(t1 // dt) // 10:])
#         r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
#                         ) / len(r[5 * int(t1 // dt) // 10:])
#
#         # Integrate recuced dynamics
#
#         redA = multi_dot([M, A, P])
#
#         if perturbation == "None":
#             hatredA = redA
#             hatLambda = redA
#         elif perturbation == "All":
#             hatredA = (multi_dot([M**2, A, P]).T / np.diag(np.dot(M, M.T))).T
#             hatLambda = multi_dot([M, A, pinv(M)])
#         elif perturbation == "hatredA":
#             hatredA = (multi_dot([M**2, A, P]).T / np.diag(np.dot(M, M.T))).T
#             hatLambda = redA
#         elif perturbation == "hatLambda":
#             hatredA = redA
#             hatLambda = multi_dot([M, A, pinv(M)])
#         else:
#             raise ValueError("Wrong perturbation name: "
#                              "The choices are : None, All,"
#                              " hatredA or hatLambda")
#
#         args_red_winfree_2D = (omega_array, sigma, N,
#                                hatredA, hatLambda)
#         red_winfree_sol = integrate_dynamics(t0, t1, dt,
#                                              reduced_winfree_2D,
#                                              redA, "zvode", Z0,
#                                              *args_red_winfree_2D)
#
#         Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
#         R1, R2 = np.absolute(Z1), np.absolute(Z2)
#         R = np.absolute(n1*Z1 + n2*Z2)/N
#
#         R1_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
#                          ) / len(R1[5 * int(t1 // dt) // 10:])
#         R2_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
#                          ) / len(R2[5 * int(t1 // dt) // 10:])
#         R_mean = np.sum(R[5 * int(t1 // dt) // 10:]
#                         ) / len(R[5 * int(t1 // dt) // 10:])
#
#         RMSE1 += RMSE(r1_mean, R1_mean)/nb_data
#         RMSE2 += RMSE(r2_mean, R2_mean)/nb_data
#         RMSE_tot += RMSE(r_mean, R_mean)/nb_data
#
#         if plot_temporal_series:
#             import matplotlib.pyplot as plt
#
#             plt.subplot(311)
#             plt.plot(r, color="k")
#             plt.plot(R, color="g")
#             plt.plot(r_mean*np.ones(len(r)), color="r")
#             plt.ylabel("$R$", fontsize=12)
#             plt.ylim([-0.02, 1.02])
#
#             plt.subplot(312)
#             plt.plot(r1, color="b")
#             plt.plot(R1, color="lightblue")
#             plt.ylabel("$R_1$", fontsize=12)
#             plt.ylim([-0.02, 1.02])
#
#             plt.subplot(313)
#             plt.plot(r2, color="r")
#             plt.plot(R2, color="y")
#             plt.ylabel("$R_2$", fontsize=12)
#             plt.ylim([-0.02, 1.02])
#             plt.xlabel("$t$", fontsize=12)
#             plt.tight_layout()
#
#             plt.show()
#
#     return RMSE1, RMSE2, RMSE_tot
