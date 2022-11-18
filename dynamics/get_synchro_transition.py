from dynamics.integrate import *
import matplotlib.pyplot as plt
from dynamics.dynamics import *
from dynamics.reduced_dynamics import *
from graphs.get_reduction_matrix_and_characteristics import *
import numpy as np
from numpy.linalg import multi_dot, pinv
import networkx as nx
import time
from tqdm import tqdm


"""
In this script, the functions get the synchronization transition for the 
complete dynamics and the reduced dynamics
"""

"""-------- Synchronization transition for a single graphs -----------------"""
"""------------------- Section III D. article ------------------------------"""


def get_synchro_transition_phase_dynamics_graphs(
        complete_dynamics, reduced_dynamics, t0, t1, dt, averaging,
        T_1, T_2, T_3, theta0, M, sigma_array, W, A,
        plot_time_series=False):
    """
    Generate data for a complete and the reduced phase dynamics
    on a particular graph. In particular, we get the global synchronization
    transition <R>_t vs sigma for the complete and the reduced dynamics.

    See synch_prediction/graphs/SBM/CVM_....py
                       or small_bipartite/CVM_....py
                       or two_triangles/CVM_....py


    :param complete_dynamics: N-dimensional dynamics function to integrate
                              with ode from scipy.integrate which is in
                              accordance with the function integrate dynamics
                              in synch_predictions/dynamics/integrate.
                              Choices: winfree, kuramoto, theta located in
                              synch_prediction/dynamics/dynamics.py
    :param reduced_dynamics: n<N-dimensional dynamics function to integrate
                             with ode from scipy.integrate which is in
                              accordance with the function integrate dynamics
                              in synch_predictions/dynamics/integrate.
                              Choices: reduced_winfree_complex,
                                       reduced_kuramoto_complex,
                                       reduced_theta_complex
                              located in synch_prediction/dynamics/dynamics.py
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param averaging: integer number X between 5 and 10. ex: 8
                       It means that the temporal series are averaged over
                       X*10% to 100% of their length
    :param T_1:(str) The first target matrix. ex.: "A" or "W" or "K"
    :param T_2:(str) The second target matrix. ex.: "A" or "W" or "K" or "None"
                     Must be different than T_1
    :param T_3:(str) The third target matrix. ex.: "A" or "W" or "K" or "None"
                     Must be different than T_2
    :param theta0: Initial conditions
    :param M: Reduction matrix
    :param sigma_array: Array of the coupling constant sigma
    :param W: Diagonal matrix of the natural frequencies
    :param A: Adjacency matrix
    :param plot_time_series: Plot time series of the global synchro observable
                              vs time for the complete and the reduced dynamics

    :return: synchro_transition_data_dictionary : A dictionary that contains
                                                  the following keys,
      Keys
    { "r",                      -> Global synchro observables
                                   of the complete dynamics
      "R",                      -> Global synchro observables
                                   of the reduced dynamics
      "r_mu_matrix",            -> Mesoscopic synchro observables
                                   of the complete dynamics
      "R_mu_matrix",            -> Mesoscopic synchro observables
                                   of the reduced dynamics
      "sigma_array", "M", "l_normalized_weights", "m", "W", "K", "A",
      "reduced_W", "reduced_K", "reduced_A", "n", "N", "theta0", T_1, T_2, T_3}
    """

    # Dynamical parameters
    omega = np.diag(W)
    z0 = np.exp(1j * theta0)
    Z0 = np.dot(M, z0)

    # Structural parameters
    n, N = np.shape(M)
    K = np.diag(np.sum(A, axis=0))
    l_weights = np.count_nonzero(M, axis=1)
    l_normalized_weights = l_weights / np.sum(l_weights)
    m = M.T @ l_normalized_weights

    # Reduction matrices obtained from the compatibility equations
    Omega = np.sum(M@W, axis=1)
    kappa = np.sum(M@A, axis=1)
    reduced_W = get_reduced_parameter_matrix(M, np.diag(omega))
    reduced_K = get_reduced_parameter_matrix(M, K)
    reduced_A = get_reduced_parameter_matrix(M, A)

    # Define dictionary
    synchro_transition_data_dictionary = {}
    r_time_averaged_array = np.zeros(len(sigma_array))
    r_mu_time_averaged_array = np.zeros((n, len(sigma_array)))
    R_time_averaged_array = np.zeros(len(sigma_array))
    R_mu_time_averaged_array = np.zeros((n, len(sigma_array)))

    # Get synchronization data
    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)

        # ----------------- Integrate complete dynamics -----------------------
        sigma = sigma_array[i]
        args_complete = (omega, sigma)
        # complete_sol = integrate_dynamics(t0, t1, dt, complete_dynamics, A,
        #                                   "dop853", theta0,
        #                                   *args_complete)
        complete_sol = \
            np.array(integrate_dopri45(t0, t1, dt, complete_dynamics,
                                       A, theta0, *args_complete))

        r_mu_array = np.zeros((len(complete_sol[:, 0]), n))
        for mu in range(n):
            r_mu = \
                np.absolute(np.sum(M[mu, :] * np.exp(1j * complete_sol),
                                   axis=1))
            r_mu_array[:, mu] = r_mu
            r_mu_mean = np.mean(r_mu[averaging*int(t1//dt)//10:])
            r_mu_time_averaged_array[mu, i] = r_mu_mean

        r = np.absolute(np.sum(m*np.exp(1j * complete_sol), axis=1))
        r_mean = np.mean(r[averaging*int(t1//dt)//10:])
        r_time_averaged_array[i] = r_mean

        # ---------------- Integrate reduced dynamics  ------------------------
        args_reduced = (reduced_W, reduced_K, sigma, N, kappa, Omega)
        # reduced_sol = integrate_dynamics(t0, t1, dt,
        #                                  reduced_dynamics,
        #                                  reduced_A, "zvode", Z0,
        #                                  *args_reduced)
        reduced_sol = \
            np.array(integrate_dopri45(t0, t1, dt, reduced_dynamics,
                                       reduced_A, Z0, *args_reduced))

        Z = np.zeros(len(reduced_sol[:, 0]))
        R_mu_array = np.zeros((len(reduced_sol[:, 0]), n))
        Phi_mu_array = np.zeros((len(reduced_sol[:, 0]), n))
        for mu in range(n):
            # R_mu = red_kuramoto_sol[:, mu]
            # Phi_mu = red_kuramoto_sol[:, mu+n]
            Z_mu = reduced_sol[:, mu]  # R_mu*np.exp(1j*Phi_mu) #

            R_mu = np.abs(Z_mu)
            R_mu_array[:, mu] = R_mu
            R_mu_mean = np.mean(R_mu[averaging*int(t1//dt)//10:])
            R_mu_time_averaged_array[mu, i] = R_mu_mean

            Phi_mu = np.angle(Z_mu)
            Phi_mu_array[:, mu] = Phi_mu

            Z = Z + l_normalized_weights[mu]*Z_mu

        R = np.absolute(Z)
        R_mean = np.mean(R[averaging*int(t1//dt)//10:])
        R_time_averaged_array[i] = R_mean

        if plot_time_series:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            plt.figure(figsize=(10, 10))

            plt.subplot(221)
            plt.plot(r, "k", zorder=1)
            plt.plot(R, "#bbbbbb", zorder=0)
            # plt.plot(r_mean*np.ones(len(r)))
            # plt.plot(R_mean*np.ones(len(R)))
            ylab = plt.ylabel("$R$")
            ylab.set_rotation(0)
            # plt.xlabel("Time $t$")

            plt.subplot(222)
            plt.plot(r_mu_array[:, 0], "k", zorder=1)
            plt.plot(R_mu_array[:, 0], "#bbbbbb", zorder=0)
            ylab = plt.ylabel("$R_1$")
            ylab.set_rotation(0)
            # plt.xlabel("Time $t$")

            plt.subplot(223)
            plt.plot(r_mu_array[:, 1], "k", zorder=1)
            plt.plot(R_mu_array[:, 1], "#bbbbbb", zorder=0)
            ylab = plt.ylabel("$R_2$")
            ylab.set_rotation(0)
            plt.xlabel("Time $t$")

            plt.subplot(224)
            plt.plot(Phi_mu_array[:, 0] - Phi_mu_array[:, 1],
                     "#bbbbbb", zorder=0)
            ylab = plt.ylabel("$\\Phi$")
            ylab.set_rotation(0)
            plt.xlabel("Time $t$")

            # plt.subplot(224)
            # for i in range(N):
            #     # plt.plot(np.cos(complete_sol[:, i]))
            #     plt.plot(complete_sol[:, i])
            # # ylab = plt.ylabel("$\\cos(\\theta_j)$")
            # ylab = plt.ylabel("$\\theta_j$")
            # ylab.set_rotation(0)
            # plt.xlabel("Time $t$")

            plt.tight_layout()

            plt.show()

    synchro_transition_data_dictionary["r"] = r_time_averaged_array.tolist()
    synchro_transition_data_dictionary["R"] = R_time_averaged_array.tolist()
    synchro_transition_data_dictionary["r_mu_matrix"] = \
        r_mu_time_averaged_array.tolist()
    synchro_transition_data_dictionary["R_mu_matrix"] = \
        R_mu_time_averaged_array.tolist()
    synchro_transition_data_dictionary["sigma_array"] = sigma_array.tolist()
    synchro_transition_data_dictionary["M"] = M.tolist()
    synchro_transition_data_dictionary["l_normalized_weights"] = \
        l_normalized_weights.tolist()
    synchro_transition_data_dictionary["m"] = m.tolist()
    synchro_transition_data_dictionary["W"] = W.tolist()
    synchro_transition_data_dictionary["K"] = K.tolist()
    synchro_transition_data_dictionary["A"] = A.tolist()
    synchro_transition_data_dictionary["reduced_W"] = reduced_W.tolist()
    synchro_transition_data_dictionary["reduced_K"] = reduced_K.tolist()
    synchro_transition_data_dictionary["reduced_A"] = reduced_A.tolist()
    synchro_transition_data_dictionary["n"] = n
    synchro_transition_data_dictionary["N"] = N
    synchro_transition_data_dictionary["theta0"] = theta0.tolist()
    synchro_transition_data_dictionary["T_1"] = T_1
    synchro_transition_data_dictionary["T_2"] = T_2
    synchro_transition_data_dictionary["T_3"] = T_3

    return synchro_transition_data_dictionary


def get_solutions_complete_phase_dynamics_graphs(
        complete_dynamics, t0, t1, dt, theta0, sigma_array, W, A):
    """
    Generate data for a complete phase dynamics on a particular graph.

    :param complete_dynamics: N-dimensional dynamics function to integrate
                              with ode from scipy.integrate which is in
                              accordance with the function integrate dynamics
                              in synch_predictions/dynamics/integrate.
                              Choices: winfree, kuramoto, theta located in
                              synch_prediction/dynamics/dynamics.py
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param theta0: Initial conditions
    :param sigma_array: Array of the coupling constant sigma
    :param W: Diagonal matrix of the natural frequencies
    :param A: Adjacency matrix

    :return: complete_sol_list: each element of the list is a matrix (list)
                                the solution in theta for all oscillators
                                according to time for a given sigma
    """
    omega = np.diag(W)

    complete_sol_list = []

    # Get synchronization data
    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)

        # ----------------- Integrate complete dynamics -----------------------
        sigma = sigma_array[i]
        args_complete = (omega, sigma)
        complete_sol = \
            np.array(integrate_dopri45(t0, t1, dt, complete_dynamics,
                                       A, theta0, *args_complete))
        complete_sol_list.append(complete_sol)

    return complete_sol_list


def get_solutions_reduced_phase_dynamics_graphs(
        reduced_dynamics, t0, t1, dt,
        theta0, M, sigma_array, W, A):
    """
    Generate data for a reduced phase dynamics
    on a particular graph.

    :param reduced_dynamics: n<N-dimensional dynamics function to integrate
                             with ode from scipy.integrate which is in
                              accordance with the function integrate dynamics
                              in synch_predictions/dynamics/integrate.
                              Choices: reduced_winfree_complex,
                                       reduced_kuramoto_complex,
                                       reduced_theta_complex
                              located in synch_prediction/dynamics/dynamics.py
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param theta0: Initial conditions
    :param M: Reduction matrix
    :param sigma_array: Array of the coupling constant sigma
    :param W: Diagonal matrix of the natural frequencies
    :param A: Adjacency matrix

    :return: complete_sol_list: each element of the list is a matrix (list)
                                the solution in Z for all oscillators
                                according to time for a given sigma
    """

    # Dynamical parameters
    z0 = np.exp(1j * theta0)
    Z0 = np.dot(M, z0)

    # Structural parameters
    N = len(A[0])     # Number of nodes
    K = np.diag(np.sum(A, axis=0))

    # Reduction matrices obtained from the compatibility equations
    Omega = np.sum(M@W, axis=1)
    kappa = np.sum(M@A, axis=1)
    reduced_W = get_reduced_parameter_matrix(M, W)
    reduced_K = get_reduced_parameter_matrix(M, K)
    reduced_A = get_reduced_parameter_matrix(M, A)

    reduced_sol_list = []
    # Get synchronization data
    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)

        sigma = sigma_array[i]

        args_reduced = (reduced_W, reduced_K, sigma, N, kappa, Omega)

        reduced_sol = \
            np.array(integrate_dopri45(t0, t1, dt, reduced_dynamics,
                                       reduced_A, Z0, *args_reduced))

        reduced_sol_list.append(reduced_sol)

    return reduced_sol_list


def measure_synchronization_complete_phase_dynamics(
        complete_dynamics_solutions, sigma_array, t1, dt, averaging, M):

    n, N = np.shape(M)
    m = global_reduction_matrix(M)

    r_time_averaged_array = np.zeros(len(sigma_array))
    r_mu_time_averaged_array = np.zeros((n, len(sigma_array)))
    for i in range(len(sigma_array)):
        complete_sol = complete_dynamics_solutions[i]
        r_mu_array = np.zeros((len(complete_sol[:, 0]), n))
        for mu in range(n):
            r_mu = \
                np.absolute(np.sum(M[mu, :] * np.exp(1j * complete_sol),
                                   axis=1))
            r_mu_array[:, mu] = r_mu
            r_mu_mean = np.mean(r_mu[averaging * int(t1 // dt) // 10:])
            r_mu_time_averaged_array[mu, i] = r_mu_mean
        
        r = np.absolute(np.sum(m * np.exp(1j * complete_sol), axis=1))
        r_mean = np.mean(r[averaging * int(t1 // dt) // 10:])
        r_time_averaged_array[i] = r_mean

    return r_time_averaged_array, r_mu_time_averaged_array


def measure_synchronization_reduced_phase_dynamics(
        reduced_dynamics_solutions, sigma_array, t1, dt, averaging, M):

    n, N = np.shape(M)
    l_weights = np.count_nonzero(M, axis=1)
    l_normalized_weights = l_weights / np.sum(l_weights)

    R_time_averaged_array = np.zeros(len(sigma_array))
    R_mu_time_averaged_array = np.zeros((n, len(sigma_array)))

    for i in range(len(sigma_array)):
        reduced_sol = reduced_dynamics_solutions[i]
        Z = np.zeros(len(reduced_sol[:, 0]))
        R_mu_array = np.zeros((len(reduced_sol[:, 0]), n))
        Phi_mu_array = np.zeros((len(reduced_sol[:, 0]), n))
        for mu in range(n):
            Z_mu = reduced_sol[:, mu]

            R_mu = np.abs(Z_mu)
            R_mu_array[:, mu] = R_mu
            R_mu_mean = np.mean(R_mu[averaging*int(t1//dt)//10:])
            R_mu_time_averaged_array[mu, i] = R_mu_mean

            Phi_mu = np.angle(Z_mu)
            Phi_mu_array[:, mu] = Phi_mu

            Z = Z + l_normalized_weights[mu]*Z_mu

        R = np.absolute(Z)
        R_mean = np.mean(R[averaging*int(t1//dt)//10:])
        R_time_averaged_array[i] = R_mean

    return R_time_averaged_array, R_mu_time_averaged_array


def get_multiple_synchro_transition_phase_dynamics_graphs(
        complete_dynamics, reduced_dynamics, t0, t1, dt, averaging,
        CVM_dictionary, targets_possibilities, theta0, sigma_array,
        plot_time_series=False):
    """
    Generate data for the complete and the reduced Kuramoto dynamics
    on a particular graph. In particular, we get the global synchronization
    transition <R>_t vs sigma for the complete and the reduced dynamics for
    different target matrices.

    See synch_prediction/graphs/SBM/CVM_....py
                       or small_bipartite/CVM_....py
                       or two_triangles/CVM_....py

    :param complete_dynamics: N-dimensional dynamics function to integrate
                              with ode from scipy.integrate which is in
                              accordance with the function integrate dynamics
                              in synch_predictions/dynamics/integrate.
                              Choices: winfree, kuramoto, theta located in
                              synch_prediction/dynamics/dynamics.py
    :param reduced_dynamics: n<N-dimensional dynamics function to integrate
                             with ode from scipy.integrate which is in
                              accordance with the function integrate dynamics
                              in synch_predictions/dynamics/integrate.
                              Choices: reduced_winfree_complex,
                                       reduced_kuramoto_complex,
                                       reduced_theta_complex
                              located in synch_prediction/dynamics/dynamics.py
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param averaging: integer number X between 5 and 10. ex: 8
                       It means that the temporal series are averaged over
                       X*10% to 100% of their length
    :param CVM_dictionary: CVM_dictionary: it is the resulting dictionary
                           from the function:
                           get_reduction_matrices_for_all_targets .
                           See the documentation of the function.
    :param targets_possibilities: List of strings equal to
                                ["W", "K", "A", "WK", "WA", "KW", "KA", "AW",
                                "AK", "WKA", "WAK", "KWA", "KAW", "AWK", "AKW"]
                                or a "sublist" of this list. ex. ["W", "A"]
    :param theta0: Initial conditions
    :param sigma_array: Array of the coupling constant sigma
    :param plot_time_series: Plot time series of the global synchro observable
                              vs time for the complete and the reduced dynamics

    :return: synchro_transition_data_dictionary : A dictionary that contains
                                                  the following keys,
      Keys
    { "r",                      -> Global synchro observables
                                   of the complete dynamics
      "R",                      -> Global synchro observables
                                   of the reduced dynamics
      "r_mu_matrix",            -> Mesoscopic synchro observables
                                   of the complete dynamics
      "R_mu_matrix",            -> Mesoscopic synchro observables
                                   of the reduced dynamics
      "sigma_array", "M", "l_normalized_weights", "m", "W", "K", "A",
      "reduced_W", "reduced_K", "reduced_A", "n", "N", "theta0", T_1, T_2, T_3}
    """

    multiple_synchro_transition_dictionary = {}

    W = np.array(CVM_dictionary["W"])
    K = np.array(CVM_dictionary["K"])
    A = np.array(CVM_dictionary["A"])

    N = len(A[0])
    n = len(np.array(CVM_dictionary["M_W"])[:, 0])
    # targets_possibilities = ["W", "K", "A", "WK", "WA", "KW", "KA", "AW",
    #                          "AK", "WKA", "WAK", "KWA", "KAW", "AWK", "AKW"]

    for targets_string in targets_possibilities:

        if len(targets_string) == 1:
            T_1, T_2, T_3 = targets_string, "None", "None"
        elif len(targets_string) == 2:
            T_1, T_2 = list(targets_string)
            T_3 = "None"
        else:
            T_1, T_2, T_3 = list(targets_string)
        M = np.array(CVM_dictionary[f"M_{targets_string}"])
        m = global_reduction_matrix(M)

        if matrix_is_positive(m) and matrix_is_normalized(m) \
                and matrix_is_positive(M) and matrix_is_normalized(M) \
                and matrix_has_rank_n(M):
            synchro_transition_data_dictionary = \
                get_synchro_transition_phase_dynamics_graphs(
                    complete_dynamics, reduced_dynamics, t0, t1, dt, averaging,
                    T_1, T_2, T_3, theta0, M, sigma_array, W, A,
                    plot_time_series=plot_time_series)
            multiple_synchro_transition_dictionary[
                "r_{}".format(targets_string)] \
                = synchro_transition_data_dictionary["r"]
            multiple_synchro_transition_dictionary[
                "R_{}".format(targets_string)] \
                = synchro_transition_data_dictionary["R"]

            multiple_synchro_transition_dictionary[
                "r_mu_matrix_{}".format(targets_string)] \
                = synchro_transition_data_dictionary["r_mu_matrix"]
            multiple_synchro_transition_dictionary[
                "R_mu_matrix_{}".format(targets_string)] \
                = synchro_transition_data_dictionary["R_mu_matrix"]
        else:
            raise ValueError("The reduction  matrix for the choice of targets "
                             "(targets_string) was not giving a good "
                             "observable. ")

    multiple_synchro_transition_dictionary["sigma_array"] = \
        sigma_array.tolist()
    multiple_synchro_transition_dictionary["W"] = W.tolist()
    multiple_synchro_transition_dictionary["K"] = K.tolist()
    multiple_synchro_transition_dictionary["A"] = A.tolist()
    multiple_synchro_transition_dictionary["n"] = n
    multiple_synchro_transition_dictionary["N"] = N
    multiple_synchro_transition_dictionary["theta0"] = theta0.tolist()

    return multiple_synchro_transition_dictionary


def get_synchro_transition_realizations_phase_dynamics_random_graph(
        realizations_dictionary, complete_dynamics, reduced_dynamics,
        t0, t1, dt, averaging, sigma_array, plot_time_series=False):
    """
    Generate data for the complete and the reduced phase dynamics on multiple
    graphs of a random graph ensemble, multiple initial conditions and
    multiple dynamical parameters omega.
    In particular, we get the global synchronization transitions <R>_t vs sigma
    for the complete and the reduced dynamics.

    :param realizations_dictionary: Contains the omega_realizations, the
                                    adjacency_matrix_realizations and the
                                    M_realizations which are all generated with
                                    the function # TODO
    :param complete_dynamics: N-dimensional dynamics function to integrate
                              with ode from scipy.integrate which is in
                              accordance with the function integrate dynamics
                              in synch_predictions/dynamics/integrate.
                              Choices: winfree, kuramoto, theta located in
                              synch_prediction/dynamics/dynamics.py
    :param reduced_dynamics: n < N-dimensional dynamics function to integrate
                             with ode from scipy.integrate which is in
                              accordance with the function integrate dynamics
                              in synch_predictions/dynamics/integrate.
                              Choices: reduced_winfree_complex,
                                       reduced_kuramoto_complex,
                                       reduced_theta_complex
                              located in synch_prediction/dynamics/dynamics.py
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param averaging: integer number X between 5 and 10. ex: 8
                       It means that the temporal series are averaged over
                       X*10% to 100% of their length
    :param sigma_array: Array of the coupling constant sigma
    :param plot_time_series: Plot time series of the global synchro observable
                              vs time for the complete and the reduced dynamics

    :return: multiple_realizations_synchro_transition_dictionary :
             A dictionary that contains the following keys:
      Keys
    { "r_list",              -> Global synchro observables
                                of the complete dynamics, each element of
                                the list is a time series for an instance
      "R_list",              -> Global synchro observables
                                of the reduced dynamics, each element of
                                the list is a time series for an instance
      "r_mu_matrix_list",    -> Mesoscopic synchro observables
                                of the complete dynamics, each element of
                                the list is a time series for an instance
      "R_mu_matrix",         -> Mesoscopic synchro observables
                                of the reduced dynamics, each element of
                                the list is a time series for an instance
      "sigma_array",         -> Coupling constant array
      "theta0"               -> Type of initial conditions (string)
       }

    """

    sigma_info = f"sigma_array = np.linspace({np.round(sigma_array[0], 3)}," \
                 f" {np.round(sigma_array[-1], 3)}, {len(sigma_array)})\n"

    multiple_realizations_synchro_transition_dictionary = \
        {"theta0": "2*np.pi*np.random.randn(N)",
         "sigma_array": sigma_info, "t0": t0, "t1": t1, "dt": dt,
         "averaging": averaging}

    # print(realizations_dictionary.keys())

    adjacency_matrix_realizations = \
        realizations_dictionary["adjacency_matrix_realizations"]
    M_realizations = \
        realizations_dictionary["M_realizations"]
    omega_realizations = \
        realizations_dictionary["omega_realizations"]
    T_1 = realizations_dictionary["T_1"]
    T_2 = realizations_dictionary["T_2"]
    T_3 = realizations_dictionary["T_3"]

    r_list = []
    R_list = []
    r_mu_matrix_list = []
    R_mu_matrix_list = []

    # Naive choice
    # N1, N2 = 150, 100
    # M = np.block([[1/N1*np.ones(N1), np.zeros(N2)],
    #               [np.zeros(N1), 1/N2*np.ones(N2)]])

    for instance_index in tqdm(range(len(adjacency_matrix_realizations))):
        M = np.array(M_realizations[instance_index])
        n, N = np.shape(M)
        A = np.array(adjacency_matrix_realizations[instance_index])
        W = np.array(np.diag(omega_realizations[instance_index]))
        theta0 = 2*np.pi*np.random.randn(N)
        synchro_transition_data_dictionary = \
            get_synchro_transition_phase_dynamics_graphs(
                complete_dynamics, reduced_dynamics, t0, t1, dt,
                averaging, T_1, T_2, T_3, theta0, M, sigma_array, W, A,
                plot_time_series=plot_time_series)
        r_list.append(
            synchro_transition_data_dictionary["r"])
        R_list.append(
            synchro_transition_data_dictionary["R"])
        r_mu_matrix_list.append(
            synchro_transition_data_dictionary["r_mu_matrix"])
        R_mu_matrix_list.append(
            synchro_transition_data_dictionary["R_mu_matrix"])

    multiple_realizations_synchro_transition_dictionary["r_list"]\
        = r_list
    multiple_realizations_synchro_transition_dictionary["R_list"]\
        = R_list
    multiple_realizations_synchro_transition_dictionary["r_mu_matrix_list"]\
        = r_mu_matrix_list
    multiple_realizations_synchro_transition_dictionary["R_mu_matrix_list"]\
        = R_mu_matrix_list

    multiple_realizations_synchro_transition_dictionary["sigma_array"] = \
        sigma_array.tolist()

    return multiple_realizations_synchro_transition_dictionary


def get_data_kuramoto_sakaguchi_one_star(sigma_array, N, alpha,
                                         t0, t1, dt, t0_red,
                                         t1_red, dt_red, averaging,
                                         plot_temporal_series=0):
    """
    Generate data for different reduced Kuramoto-Sakaguchi dynamics
    on a star graph

    IMPORTANT: The initial conditions of the complete and the reduced dynamics
    are different to simplify the convergence to equilibrium.

    See Chen 2017 Frontiers Physics Order parameter...
    for details on initial conditions and the complete bifurcation diagram.

    See Gomez-Gardenes 2011 PRL to know how to integrate the system and get the
    complete hysteresis.

    This function was updated in 2020-02-04.

    :param sigma_array: Array of the coupling constant sigma
    :param N: Number of nodes
    :param alpha: Phase-lag between the oscillators' phases
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
                               sigma_array,    sigma_array,
                               omega_array,    omega_array
                               alpha,              alpha,
                                N,                  N    }
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
    # omega = np.array([omega1] + n2 * [omega2])
    Omega = (omega1 + (N-1)*omega2)/N
    omega_CM = np.array([omega1-Omega] + n2 * [omega2-Omega])

    # print(sigma_critical(omega1, omega2, N, alpha))

    """" Get the bottom branch (forward branch) """
    theta0 = np.linspace(0, 2*np.pi*(1-1/N), N)
    # theta0 = np.linspace(0, np.pi//2, N)
    Zp = np.mean(np.exp(1j*theta0[1:]))
    Rp = np.absolute(Zp)
    Phip = np.angle(Zp)
    Phic = theta0[0]
    W0 = [Rp, Phic - Phip]
    # W03 = [Rp, Phic, Phip]
    # print(np.absolute(np.exp(1j*W0[1]) + n2*W0[0]) / N)

    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)

        sigma = sigma_array[i]

        # Integrate complete dynamics

        args_kuramoto = (omega_CM, N*sigma, alpha)
        # *N, to cancel the N in the definition of the dynamics "kuramoto"
        # in my code.
        kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto_sakaguchi, A,
                                          "dop853", theta0,
                                          *args_kuramoto)

        r2 = np.absolute(
            np.sum(M[1, :] * np.exp(1j * kuramoto_sol),
                   axis=1))
        Z_complete = np.mean(
            (n1*M[0, :] + n2*M[1, :])*np.exp(1j*kuramoto_sol), axis=1)
        r = np.absolute(Z_complete)
        # psi = np.angle(Z_complete)
        diff_core_periph = kuramoto_sol.T[0, :] - kuramoto_sol.T
        phi = np.mean(diff_core_periph.T[:, 1:], axis=1)

        r2_mean = np.sum(r2[averaging * int(t1 // dt) // 10:]
                         ) / len(r2[averaging * int(t1 // dt) // 10:])
        r_mean = np.sum(r[averaging * int(t1 // dt) // 10:]
                        ) / len(r[averaging * int(t1 // dt) // 10:])

        # Integrate reduced dynamics

        MAMp = multi_dot([M, A, pinv(M)])
        args_red_kuramoto_sakaguchi = (N, omega1, omega2, sigma, alpha)
        """ 2D : R, Phi = Ph1-Phi2 """
        red_kuramoto_sol = \
            integrate_dynamics(t0_red, t1_red, dt_red,
                               reduced_kuramoto_sakaguchi_star_2D,
                               MAMp, "dop853", W0,
                               *args_red_kuramoto_sakaguchi)
        R2 = red_kuramoto_sol[:, 0]
        Phi = red_kuramoto_sol[:, 1]

        """ 3D : R, Phi1, Phi2: You need to integrate the 3D reduced dyn """
        # red_kuramoto_sol = \
        #     integrate_dynamics(t0_red, t1_red, dt_red,
        #                        reduced_kuramoto_sakaguchi_star_3D,
        #                        MAMp, "dop853", W03,
        #                        *args_red_kuramoto_sakaguchi)
        #
        # R2 = red_kuramoto_sol[:, 0]
        # Phi1 = red_kuramoto_sol[:, 1]
        # Phi2 = red_kuramoto_sol[:, 2]
        # Z = (np.exp(1j*Phi1) + n2*R2*np.exp(1j*Phi2)) / N
        # Rpsi = np.absolute(Z)
        # Psi = np.angle(Z)
        # Phi = Phi1 - Phi2

        R = np.absolute(np.exp(1j*Phi) + n2*R2) / N

        R2_mean = np.sum(R2[averaging * int(t1 // dt) // 10:]
                         )/len(R2[averaging*int(t1//dt)//10:])
        # Phi_mean = np.sum(Phi[averaging * int(t1 // dt) // 10:]
        #                   ) / len(Phi[averaging * int(t1 // dt) // 10:])
        R_mean = np.sum(R[averaging*int(t1//dt)//10:]
                        )/len(R[averaging*int(t1//dt)//10:])

        r_matrix[i, 0] = r_mean
        r2_matrix[i, 0] = r2_mean

        R_matrix[i, 0] = R_mean
        R2_matrix[i, 0] = R2_mean

        theta0 = kuramoto_sol[-1, :]
        Zp = np.mean(np.exp(1j * theta0[1:]))
        Rp = np.absolute(Zp)
        Phip = np.angle(Zp)
        Phic = theta0[0]
        W0 = [Rp, Phic - Phip]

        if plot_temporal_series:
            # import matplotlib.pyplot as plt

            """ Complete dynamics"""
            """
            first_community_color = "#2171b5"
            reduced_first_community_color = "#9ecae1"
            total_color = "#525252"
            fontsize_legend = 12
            labelsize = 12

            fig = plt.figure(figsize=(3, 3))
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.plot(r * np.cos(phi), r * np.sin(phi), linewidth=1,
                     color=first_community_color)
            plt.xlabel("$R(t) \\cos\\Phi(t)$", fontsize=12)
            plt.ylabel("$R(t) \\sin\\Phi(t)$", fontsize=12)
            # plt.xticks([-0.2, 0, 0.2])
            # plt.yticks([-0.2, 0, 0.2])
            plt.tick_params(axis='both', which='major', labelsize=labelsize)
            plt.tight_layout()
            plt.show()
            """

            """ Reduced dynamics """
            """
            plt.figure(figsize=(5, 5))
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.plot(R2 * np.cos(Phi), R2 * np.sin(Phi))
            plt.xlabel("$R_2(t) \\cos\\Phi(t)$", fontsize=12)
            plt.ylabel("$R_2(t) \\sin\\Phi(t)$", fontsize=12)
            plt.show()
            """

            """ Phi , Psi """
            """
            first_community_color = "#2171b5"
            reduced_first_community_color = "#9ecae1"
            total_color = "#525252"
            fontsize_legend = 12
            labelsize = 12

            fig = plt.figure(figsize=(5, 8))
            time_array = np.arange(t0, t1, dt)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            ax = plt.subplot(321)
            ax.plot(time_array, r, first_community_color,
                    label="Complete dynamics")
            ax.plot(time_array, R, reduced_first_community_color,
                    label="Reduced dynamics")
            ax.plot(time_array, Rpsi, "#222222")
            ax.set_xlabel("Time $t$", fontsize=12)
            ax.set_ylabel("$R(t)$", fontsize=12)
            # ax.set_xlim([96, 100])
            # ax.set_ylim([0.117, 0.163])
            # ax.set_xticks([96, 98, 100])
            # plt.yticks([0.12, 0.14, 0.16])
            ax.tick_params(axis='both', which='major', labelsize=labelsize)

            # plt.legend(bbox_to_anchor=(1, 1, 1, 1),
            #            ncol=2, fontsize=fontsize_legend)

            # plt.subplot(224)
            # plt.plot(psi)
            # plt.xlabel("$t$", fontsize=12)
            # plt.ylabel("$\\Psi(t)$", fontsize=12)

            plt.subplot(322)
            plt.plot(time_array, np.sin(phi), first_community_color)
            plt.plot(time_array, np.sin(psi), "#bbbbbb")
            plt.plot(time_array, np.sin(Phi), reduced_first_community_color)
            plt.plot(time_array, np.sin(Psi), "#222222")
            plt.xlabel("Time $t$", fontsize=12)
            plt.ylabel("$\\sin\\Phi(t)$", fontsize=12)
            # plt.xlim([96, 100])
            # plt.ylim([-1.1, 1.1])
            # ax.set_xticks([96, 98, 100])
            # plt.yticks([-1, 0, 1])
            plt.tick_params(axis='both', which='major', labelsize=labelsize)

            plt.subplot(323)
            plt.plot(r * np.cos(psi), r * np.sin(psi), linewidth=1,
                     color=first_community_color)
            plt.scatter(r[-1] * np.cos(psi[-1]), r[-1] * np.sin(psi[-1]),
                        s=100,
                        color=total_color, zorder=1)
            plt.xlabel("$R(t) \\cos\\Psi(t)$", fontsize=12)
            plt.ylabel("$R(t) \\sin\\Psi(t)$", fontsize=12)
            # plt.xticks([-0.2, 0, 0.2])
            # plt.yticks([-0.2, 0, 0.2])
            plt.tick_params(axis='both', which='major', labelsize=labelsize)

            plt.subplot(324)
            plt.plot(Rpsi * np.cos(Psi), Rpsi * np.sin(Psi), linewidth=1,
                     color=reduced_first_community_color, zorder=0)
            plt.scatter(Rpsi[-1] * np.cos(Psi[-1]), Rpsi[-1] * np.sin(Psi[-1]),
                        s=100,
                        color=total_color, zorder=1)
            plt.xlabel("$R(t) \\cos\\Psi(t)$", fontsize=12)
            plt.ylabel("$R(t) \\sin\\Psi(t)$", fontsize=12)
            # plt.xticks([-0.2, 0, 0.2])
            # plt.yticks([-0.2, 0, 0.2])
            plt.tick_params(axis='both', which='major', labelsize=labelsize)

            plt.subplot(325)
            plt.plot(r * np.cos(phi), r * np.sin(phi), linewidth=1,
                     color=first_community_color)
            plt.xlabel("$R(t) \\cos\\Phi(t)$", fontsize=12)
            plt.ylabel("$R(t) \\sin\\Phi(t)$", fontsize=12)
            # plt.xticks([-0.2, 0, 0.2])
            # plt.yticks([-0.2, 0, 0.2])
            plt.tick_params(axis='both', which='major', labelsize=labelsize)

            plt.subplot(326)
            plt.plot(R * np.cos(Phi), R * np.sin(Phi), linewidth=1,
                     color=reduced_first_community_color, zorder=0)
            plt.scatter(R[-1] * np.cos(Phi[-1]), R[-1] * np.sin(Phi[-1]),
                        s=100,
                        color=total_color, zorder=1)
            plt.xlabel("$R(t) \\cos\\Phi(t)$", fontsize=12)
            plt.ylabel("$R(t) \\sin\\Phi(t)$", fontsize=12)
            # plt.xticks([-0.2, 0, 0.2])
            # plt.yticks([-0.2, 0, 0.2])
            plt.tick_params(axis='both', which='major', labelsize=labelsize)

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc=(0.14, 0.9),
                       ncol=2, fontsize=fontsize_legend)
            # plt.tight_layout()
            plt.subplots_adjust(top=0.87,
                                bottom=0.11,
                                left=0.14,
                                right=0.97,
                                hspace=0.445,
                                wspace=0.525)
            plt.show()
            """

            """ Nice plots in different "space" """
            """
            first_community_color = "#2171b5"
            reduced_first_community_color = "#9ecae1"
            total_color = "#525252"
            fontsize_legend = 12
            labelsize = 12

            fig = plt.figure(figsize=(5, 5))
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            time_array = np.arange(t0, t1, dt)

            ax = plt.subplot(221)
            ax.plot(time_array, r, first_community_color,
                    label="Complete dynamics")
            ax.plot(time_array, R, reduced_first_community_color,
                    label="Reduced dynamics")
            ax.set_xlabel("Time $t$", fontsize=12)
            ax.set_ylabel("$R(t)$", fontsize=12)
            # ax.set_xlim([96, 100])
            # ax.set_ylim([0.117, 0.163])
            # ax.set_xticks([96, 98, 100])
            # plt.yticks([0.12, 0.14, 0.16])
            ax.tick_params(axis='both', which='major', labelsize=labelsize)

            # plt.legend(bbox_to_anchor=(1, 1, 1, 1),
            #            ncol=2, fontsize=fontsize_legend)

            # plt.subplot(224)
            # plt.plot(psi)
            # plt.xlabel("$t$", fontsize=12)
            # plt.ylabel("$\\Psi(t)$", fontsize=12)

            plt.subplot(222)
            plt.plot(time_array, np.sin(phi), first_community_color)
            plt.plot(time_array, np.sin(Phi), reduced_first_community_color)
            plt.xlabel("Time $t$", fontsize=12)
            plt.ylabel("$\\sin\\Phi(t)$", fontsize=12)
            # plt.xlim([96, 100])
            # plt.ylim([-1.1, 1.1])
            # ax.set_xticks([96, 98, 100])
            # plt.yticks([-1, 0, 1])
            plt.tick_params(axis='both', which='major', labelsize=labelsize)

            plt.subplot(223)
            plt.plot(r * np.cos(phi), r * np.sin(phi), linewidth=1,
                     color=first_community_color)
            plt.xlabel("$R(t) \\cos\\Phi(t)$", fontsize=12)
            plt.ylabel("$R(t) \\sin\\Phi(t)$", fontsize=12)
            # plt.xticks([-0.2, 0, 0.2])
            # plt.yticks([-0.2, 0, 0.2])
            plt.tick_params(axis='both', which='major', labelsize=labelsize)

            plt.subplot(224)
            plt.plot(R * np.cos(Phi), R * np.sin(Phi), linewidth=1,
                     color=reduced_first_community_color, zorder=0)
            plt.scatter(R[-1] * np.cos(Phi[-1]), R[-1] * np.sin(Phi[-1]),
                        marker="*", s=300, zorder=1,
                        color=total_color, edgecolors='w')
            plt.xlabel("$R(t) \\cos\\Phi(t)$", fontsize=12)
            plt.ylabel("$R(t) \\sin\\Phi(t)$", fontsize=12)
            # plt.xticks([-0.2, 0, 0.2])
            # plt.yticks([-0.2, 0, 0.2])
            plt.tick_params(axis='both', which='major', labelsize=labelsize)

            # plt.subplot(222)
            # theta_core = kuramoto_sol[:, 0]
            # for i in range(N):
            #     plt.plot(np.cos(theta_core - kuramoto_sol[:, i]))
            # plt.xlabel("$t$", fontsize=12)
            # plt.ylabel("$\\cos(\\theta^c - \\theta_j^p)$", fontsize=12)

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc=(0.14, 0.9),
                       ncol=2, fontsize=fontsize_legend)
            # plt.tight_layout()
            plt.subplots_adjust(top=0.87,
                                bottom=0.11,
                                left=0.14,
                                right=0.97,
                                hspace=0.445,
                                wspace=0.525)
            plt.show()


            save_temporal_serie = False

            if save_temporal_serie:

                with open(f'data/kuramoto_sakaguchi/'                          
                          f'r_vs_t_kuramoto_sakaguchi_star_2D_N_{N}'
                          f'_alpha_0_2pi_sigma_{sigma}.json', 'w') as outfile:
                    json.dump(r.tolist(), outfile)

                with open(f'data/kuramoto_sakaguchi/'                          
                          f'phi_vs_t_kuramoto_sakaguchi_star_2D_N_{N}'
                          f'_alpha_0_2pi_sigma_{sigma}.json', 'w') as outfile:
                    json.dump(phi.tolist(), outfile)

                with open(f'data/kuramoto_sakaguchi/'                         
                          f'R_vs_t_red_kuramoto_sakaguchi_star_2D_N_{N}'           
                          f'_alpha_0_2pi_sigma_{sigma}.json', 'w') as outfile:
                    json.dump(R.tolist(), outfile)

                with open(f'data/kuramoto_sakaguchi/'                             
                          f'Phi_vs_t_red_kuramoto_sakaguchi_star_2D_N_{N}'             
                          f'_alpha_0_2pi_sigma_{sigma}.json', 'w') as outfile:
                    json.dump(Phi.tolist(), outfile)

            """

            """ Compare R for reduced and complete dynamics"""
            """
            plt.figure(figsize=(8, 8))
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            second_community_color = "#f16913"
            reduced_second_community_color = "#fdd0a2"
            ylim = [-0.02, 1.1]
            
            plt.subplot(211)
            plt.suptitle("$\\sigma = {}, \\omega_1 = {}, \\omega_2 = {}$"
                         .format(np.round(sigma_array[i], 3),
                                 np.round(omega1, 3),
                                 np.round(omega2, 3)), y=1.0)
            plt.plot(r, color="k", label="Complete spectral")
            plt.plot(R, color="grey", label="Reduced spectral")
            plt.plot(r_mean*np.ones(int(t1//dt)), color="r")
            plt.plot(R_mean*np.ones(int(t1//dt)), color="orange")
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
            """

            """ Try animation ... """
            # import matplotlib.animation as animation
            #
            # # First set up the figure, the axis, and the plot element
            # we want to animate
            # fig = plt.figure()
            # ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
            # line, = ax.plot([], [], lw=2)
            #
            # # initialization function: plot the background of each frame
            # def init():
            #     # line.set_data([], [])
            #     line.set_data([], [])
            #     return line,
            #
            # # animation function.  This is called sequentially
            # def animate(i):
            #     x = np.arange(0, len(kuramoto_sol[:, 0]))
            #  np.cos(kuramoto_sol[i, 0])
            #     y = np.sin(kuramoto_sol[i, 0])
            #     line.set_data(x, y)
            #     return line,
            #
            # # call the animator.
            # #  blit=True means only re-draw the parts that have changed.
            # anim = animation.FuncAnimation(fig, animate, init_func=init,
            #                                frames=50, interval=20, blit=True)
            # plt.show()

    """ Get the top branch (backward branch) """
    theta0 = np.linspace(0, 2 * np.pi * (1 - 1 / N), N)
    Zp = np.mean(np.exp(1j * theta0[1:]))
    Rp = np.absolute(Zp)
    Phip = np.angle(Zp)
    Phic = theta0[0]
    W0 = [Rp, Phic - Phip]

    # print(W0)

    for k in tqdm(range(len(sigma_array))):
        time.sleep(1)

        sigma = sigma_array[-1 - k]

        # Integrate complete dynamics

        args_kuramoto = (omega_CM, N*sigma, alpha)
        # *N, to cancel the N in the definition of the dynamics "kuramoto"
        # in my code.
        kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto_sakaguchi, A,
                                          "dop853", theta0,
                                          *args_kuramoto)

        r2 = np.absolute(
            np.sum(M[1, :] * np.exp(1j * kuramoto_sol),
                   axis=1))
        r = np.absolute(
            np.sum(
                (n1 * M[0, :] + n2 * M[1, :]) * np.exp(1j * kuramoto_sol),
                axis=1)) / N

        r2_mean_top = np.sum(r2[averaging * int(t1 // dt) // 10:]
                             ) / len(r2[averaging * int(t1 // dt) // 10:])
        r_mean_top = np.sum(r[averaging * int(t1 // dt) // 10:]
                            ) / len(r[averaging * int(t1 // dt) // 10:])

        # Integrate reduced dynamics

        MAMp = multi_dot([M, A, pinv(M)])
        args_red_kuramoto_sakaguchi = (N, omega1, omega2, sigma, alpha)
        red_kuramoto_sol = \
            integrate_dynamics(t0_red, t1_red, dt_red,
                               reduced_kuramoto_sakaguchi_star_2D,
                               MAMp, "dop853", W0,
                               *args_red_kuramoto_sakaguchi)

        R2 = red_kuramoto_sol[:, 0]
        Phi = red_kuramoto_sol[:, 1]

        R = np.absolute(np.exp(1j * Phi) + n2 * R2) / N

        R2_mean_top = np.sum(R2[averaging * int(t1 // dt) // 10:]
                             ) / len(R2[averaging * int(t1 // dt) // 10:])
        # Phi_mean = np.sum(Phi[averaging * int(t1 // dt) // 10:]
        #                   ) / len(Phi[averaging * int(t1 // dt) // 10:])
        R_mean_top = np.sum(R[averaging * int(t1 // dt) // 10:]
                            ) / len(R[averaging * int(t1 // dt) // 10:])

        r_matrix[-1-k, 1] = r_mean_top
        r2_matrix[-1-k, 1] = r2_mean_top

        R_matrix[-1-k, 1] = R_mean_top
        R2_matrix[-1-k, 1] = R2_mean_top

        theta0 = kuramoto_sol[-1, :]
        Zp = np.mean(np.exp(1j * theta0[1:]))
        Rp = np.absolute(Zp)
        Phip = np.angle(Zp)
        Phic = theta0[0]
        W0 = [Rp, Phic - Phip]

        if plot_temporal_series:
            # import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 8))

            second_community_color = "#f16913"
            reduced_second_community_color = "#fdd0a2"
            ylim = [-0.02, 1.1]

            plt.subplot(211)
            plt.suptitle("$\\sigma = {}, \\omega_1 = {}, \\omega_2 = {}$"
                         .format(np.round(sigma_array[-1 - k], 3),
                                 np.round(omega1, 3),
                                 np.round(omega2, 3)), y=1.0)
            plt.plot(r, color="k", label="Complete spectral")
            plt.plot(R, color="grey", label="Reduced spectral")
            # plt.plot(r_mean * np.ones(int(t1 // dt)), color="r")
            # plt.plot(R_mean * np.ones(int(t1 // dt)), color="orange")
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
    R_dictionary["alpha"] = alpha
    R_dictionary["N"] = N

    return R_dictionary


def get_synchro_transition_phase_dynamics_graphs_backward_forward(
        complete_dynamics, reduced_dynamics, t0, t1, dt, averaging,
        T_1, T_2, T_3, theta0, M, sigma_array, W, A,
        plot_time_series=False):
    """
    Generate data for a complete and the reduced phase dynamics
    on a particular graph. In particular, we get the global synchronization
    transition <R>_t vs sigma for the complete and the reduced dynamics.

    See synch_prediction/graphs/SBM/CVM_....py
                       or small_bipartite/CVM_....py
                       or two_triangles/CVM_....py


    :param complete_dynamics: N-dimensional dynamics function to integrate
                              with ode from scipy.integrate which is in
                              accordance with the function integrate dynamics
                              in synch_predictions/dynamics/integrate.

                              Choices: winfree, kuramoto, theta located in
                              synch_prediction/dynamics/dynamics.py

    :param reduced_dynamics: n<N-dimensional dynamics function to integrate
                             with ode from scipy.integrate which is in
                              accordance with the function integrate dynamics
                              in synch_predictions/dynamics/integrate.

                              Choices: reduced_winfree_complex,
                                       reduced_kuramoto_complex,
                                       reduced_theta_complex
                              located in
                              synch_prediction/dynamics/reduced_dynamics.py
    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param averaging: integer number X between 5 and 10. ex: 8
                       It means that the temporal series are averaged over
                       X*10% to 100% of their length
    :param T_1:(str) The first target matrix. ex.: "A" or "W" or "K"
    :param T_2:(str) The second target matrix. ex.: "A" or "W" or "K" or "None"
                     Must be different than T_1
    :param T_3:(str) The third target matrix. ex.: "A" or "W" or "K" or "None"
                     Must be different than T_2
    :param theta0: Initial conditions
    :param M: Reduction matrix
    :param sigma_array: Array of the coupling constant sigma
    :param W: Diagonal matrix of the natural frequencies
    :param A: Adjacency matrix
    :param plot_time_series: Plot time series of the global synchro observable
                              vs time for the complete and the reduced dynamics

    :return: synchro_transition_data_dictionary : A dictionary that contains
                                                  the following keys,
      Keys
    { "r",                      -> Global synchro observables
                                   of the complete dynamics
      "R",                      -> Global synchro observables
                                   of the reduced dynamics
      "r_mu_matrix",            -> Mesoscopic synchro observables
                                   of the complete dynamics
      "R_mu_matrix",            -> Mesoscopic synchro observables
                                   of the reduced dynamics
      "sigma_array", "M", "l_normalized_weights", "m", "W", "K", "A",
      "reduced_W", "reduced_K", "reduced_A", "n", "N", "theta0", T_1, T_2, T_3}
    """

    # Dynamical parameters
    omega = np.diag(W)
    z0 = np.exp(1j * theta0)
    Z0 = np.dot(M, z0)

    # Structural parameters
    N = len(A[0])     # Number of nodes
    K = np.diag(np.sum(A, axis=0))
    n = len(M[:, 0])  # Number of dimension of the reduced dynamics
    l_weights = np.count_nonzero(M, axis=1)
    l_normalized_weights = l_weights / np.sum(l_weights)
    m = M.T @ l_normalized_weights

    # Reduction matrices obtained from the compatibility equations
    Omega = np.sum(M@W, axis=1)
    kappa = np.sum(M@A, axis=1)
    reduced_W = get_reduced_parameter_matrix(M, np.diag(omega))
    reduced_K = get_reduced_parameter_matrix(M, K)
    reduced_A = get_reduced_parameter_matrix(M, A)

    # Define dictionary
    synchro_transition_data_dictionary = {}

    # Get synchronization data
    for j in range(2):
        if j:
            branch_str = "forward"
            sig_array = sigma_array
            r_time_averaged_array = np.zeros(len(sigma_array))
            r_mu_time_averaged_array = np.zeros((n, len(sigma_array)))
            R_time_averaged_array = np.zeros(len(sigma_array))
            R_mu_time_averaged_array = np.zeros((n, len(sigma_array)))

        else:
            branch_str = "backward"
            sig_array = sigma_array[::-1]
            r_time_averaged_array = np.zeros(len(sigma_array))
            r_mu_time_averaged_array = np.zeros((n, len(sigma_array)))
            R_time_averaged_array = np.zeros(len(sigma_array))
            R_mu_time_averaged_array = np.zeros((n, len(sigma_array)))

        for i in tqdm(range(len(sig_array))):
            time.sleep(1)

            # --------------- Integrate complete dynamics ---------------------
            sigma = sigma_array[i]
            args_complete = (omega, sigma)
            # complete_sol = integrate_dynamics(t0, t1, dt, complete_dynamics,
            #                                   A,
            #                                   "dop853", theta0,
            #                                   *args_complete)
            complete_sol = \
                np.array(integrate_dopri45(t0, t1, dt, complete_dynamics,
                                           A, theta0, *args_complete))

            for mu in range(n):
                r_mu = \
                    np.absolute(np.sum(M[mu, :] * np.exp(1j * complete_sol),
                                       axis=1))
                r_mu_mean = np.mean(r_mu[averaging*int(t1//dt)//10:])
                r_mu_time_averaged_array[mu, i] = r_mu_mean

            r = np.absolute(np.sum(m*np.exp(1j * complete_sol), axis=1))
            r_mean = np.mean(r[averaging*int(t1//dt)//10:])
            r_time_averaged_array[i] = r_mean

            # -------------- Integrate reduced dynamics  ----------------------
            args_reduced = (reduced_W, reduced_K, sigma, N, kappa, Omega)
            # reduced_sol = integrate_dynamics(t0, t1, dt,
            #                                  reduced_dynamics,
            #                                  reduced_A, "zvode", Z0,
            #                                  *args_reduced)
            reduced_sol = \
                np.array(integrate_dopri45(t0, t1, dt, reduced_dynamics,
                                           reduced_A, Z0, *args_reduced))

            Z = np.zeros(len(reduced_sol[:, 0]))
            for mu in range(n):
                # R_mu = red_kuramoto_sol[:, mu]
                # Phi_mu = red_kuramoto_sol[:, mu+n]
                Z_mu = reduced_sol[:, mu]  # R_mu*np.exp(1j*Phi_mu) #
                R_mu = np.abs(Z_mu)
                R_mu_mean = np.mean(R_mu[averaging*int(t1//dt)//10:])
                R_mu_time_averaged_array[mu, i] = R_mu_mean
                Z = Z + l_normalized_weights[mu]*Z_mu

            R = np.absolute(Z)
            R_mean = np.mean(R[averaging*int(t1//dt)//10:])
            R_time_averaged_array[i] = R_mean

            if plot_time_series:
                plt.plot(r, "k", zorder=1)
                plt.plot(R, "#bbbbbb", zorder=0)
                plt.plot(r_mean*np.ones(len(r)))
                plt.plot(R_mean*np.ones(len(R)))
                plt.show()

        synchro_transition_data_dictionary[f"r_{branch_str}"] = \
            r_time_averaged_array.tolist()
        synchro_transition_data_dictionary[f"R_{branch_str}"] = \
            R_time_averaged_array.tolist()
        synchro_transition_data_dictionary[f"r_mu_matrix_{branch_str}"] = \
            r_mu_time_averaged_array.tolist()
        synchro_transition_data_dictionary[f"R_mu_matrix_{branch_str}"] = \
            R_mu_time_averaged_array.tolist()

    synchro_transition_data_dictionary["sigma_array"] = sigma_array.tolist()
    synchro_transition_data_dictionary["M"] = M.tolist()
    synchro_transition_data_dictionary["l_normalized_weights"] = \
        l_normalized_weights.tolist()
    synchro_transition_data_dictionary["m"] = m.tolist()
    synchro_transition_data_dictionary["W"] = W.tolist()
    synchro_transition_data_dictionary["K"] = K.tolist()
    synchro_transition_data_dictionary["A"] = A.tolist()
    synchro_transition_data_dictionary["reduced_W"] = reduced_W.tolist()
    synchro_transition_data_dictionary["reduced_K"] = reduced_K.tolist()
    synchro_transition_data_dictionary["reduced_A"] = reduced_A.tolist()
    synchro_transition_data_dictionary["n"] = n
    synchro_transition_data_dictionary["N"] = N
    synchro_transition_data_dictionary["theta0"] = theta0.tolist()
    synchro_transition_data_dictionary["T_1"] = T_1
    synchro_transition_data_dictionary["T_2"] = T_2
    synchro_transition_data_dictionary["T_3"] = T_3

    return synchro_transition_data_dictionary


# ------------------------- Old -----------------------------------------------


def get_synchro_transition_kuramoto_graphs(t0, t1, dt, averaging,
                                           T_1, T_2, T_3, theta0, M,
                                           sigma_array, W, A,
                                           plot_time_series=False):
    """
    Generate data for the complete and the reduced Kuramoto dynamics
    on a particular graph. In particular, we get the global synchronization
    transition <R>_t vs sigma for the complete and the reduced dynamics.

    See synch_prediction/graphs/SBM/CVM_....py
                       or small_bipartite/CVM_....py
                       or two_triangles/CVM_....py

    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param averaging: integer number X between 5 and 10. ex: 8
                       It means that the temporal series are averaged over
                       X*10% to 100% of their length
    :param T_1:(str) The first target matrix. ex.: "A" or "W" or "K"
    :param T_2:(str) The second target matrix. ex.: "A" or "W" or "K" or "None"
                     Must be different than T_1
    :param T_3:(str) The third target matrix. ex.: "A" or "W" or "K" or "None"
                     Must be different than T_2
    :param theta0: Initial conditions
    :param M: Reduction matrix
    :param sigma_array: Array of the coupling constant sigma
    :param W: Diagonal matrix of the natural frequencies
    :param A: Adjacency matrix
    :param plot_time_series: Plot time series of the global synchro observable
                              vs time for the complete and the reduced dynamics

    :return: synchro_transition_data_dictionary : A dictionary that contains
                                                  the following keys,
      Keys
    { "r",                      -> Global synchro observables
                                   of the complete dynamics
      "R",                      -> Global synchro observables
                                   of the reduced dynamics
      "r_mu_matrix",            -> Mesoscopic synchro observables
                                   of the complete dynamics
      "R_mu_matrix",            -> Mesoscopic synchro observables
                                   of the reduced dynamics
      "sigma_array", "M", "l_normalized_weights", "m", "W", "K", "A",
      "reduced_W", "reduced_K", "reduced_A", "n", "N", "theta0", T_1, T_2, T_3}
    """

    # Dynamical parameters
    omega = np.diag(W)
    z0 = np.exp(1j * theta0)
    Z0 = np.dot(M, z0)
    # R0 = np.absolute(Z0)
    # Phi0 = np.angle(Z0)
    # W0 = np.concatenate([R0, Phi0])

    # Structural parameters
    N = len(A[0])  # Number of nodes
    K = np.diag(np.sum(A, axis=0))
    n = len(M[:, 0])  # Number of dimension of the reduced dynamics
    l_weights = np.count_nonzero(M, axis=1)
    l_normalized_weights = l_weights / np.sum(l_weights)
    m = M.T @ l_normalized_weights

    # Reduction matrices obtained from the compatibility equations
    kappa = np.sum(np.dot(M, A), axis=1)
    reduced_W = get_reduced_parameter_matrix(M, np.diag(omega))
    reduced_K = get_reduced_parameter_matrix(M, K)
    reduced_A = get_reduced_parameter_matrix(M, A)

    # Define dictionary
    synchro_transition_data_dictionary = {}
    r_time_averaged_array = np.zeros(len(sigma_array))
    r_mu_time_averaged_array = np.zeros((n, len(sigma_array)))
    R_time_averaged_array = np.zeros(len(sigma_array))
    R_mu_time_averaged_array = np.zeros((n, len(sigma_array)))

    # Get synchronization data
    for i in tqdm(range(len(sigma_array))):
        time.sleep(1)

        # ----------------- Integrate complete dynamics -----------------------
        args_kuramoto = (omega, sigma_array[i])
        kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                          "dop853", theta0,
                                          *args_kuramoto)
        for mu in range(n):
            r_mu = \
                np.absolute(np.sum(M[mu, :] * np.exp(1j * kuramoto_sol),
                                   axis=1))
            r_mu_mean = np.mean(r_mu[averaging*int(t1//dt)//10:])
            r_mu_time_averaged_array[mu, i] = r_mu_mean

        r = np.absolute(np.sum(m*np.exp(1j * kuramoto_sol), axis=1))
        r_mean = np.mean(r[averaging*int(t1//dt)//10:])
        r_time_averaged_array[i] = r_mean

        # ---------------- Integrate reduced dynamics  ------------------------
        # """
        args_red_kuramoto_2D = (reduced_W, reduced_K, sigma_array[i],
                                N, kappa)
        red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                              reduced_kuramoto_complex,
                                              reduced_A, "zvode", Z0,
                                              *args_red_kuramoto_2D)
        # """
        """
        args_red_kuramoto = (reduced_W, reduced_K, sigma_array[i], N, kappa)
        red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                              reduced_kuramoto_R_Phi,
                                              reduced_A, "dop853", W0,
                                              *args_red_kuramoto)
        """
        Z = np.zeros(len(red_kuramoto_sol[:, 0]))
        for mu in range(n):
            # R_mu = red_kuramoto_sol[:, mu]
            # Phi_mu = red_kuramoto_sol[:, mu+n]
            Z_mu = red_kuramoto_sol[:, mu]  # R_mu*np.exp(1j*Phi_mu) #
            R_mu = np.abs(Z_mu)
            R_mu_mean = np.mean(R_mu[averaging*int(t1//dt)//10:])
            R_mu_time_averaged_array[mu, i] = R_mu_mean
            Z = Z + l_normalized_weights[mu]*Z_mu

        R = np.absolute(Z)
        R_mean = np.mean(R[averaging*int(t1//dt)//10:])
        R_time_averaged_array[i] = R_mean

        if plot_time_series:
            plt.plot(r, "k", zorder=1)
            plt.plot(R, "#bbbbbb", zorder=0)
            plt.plot(r_mean*np.ones(len(r)))
            plt.plot(R_mean*np.ones(len(R)))
            plt.show()

    synchro_transition_data_dictionary["r"] = r_time_averaged_array.tolist()
    synchro_transition_data_dictionary["R"] = R_time_averaged_array.tolist()
    synchro_transition_data_dictionary["r_mu_matrix"] = \
        r_mu_time_averaged_array.tolist()
    synchro_transition_data_dictionary["R_mu_matrix"] = \
        R_mu_time_averaged_array.tolist()
    synchro_transition_data_dictionary["sigma_array"] = sigma_array.tolist()
    synchro_transition_data_dictionary["M"] = M.tolist()
    synchro_transition_data_dictionary["l_normalized_weights"] = \
        l_normalized_weights.tolist()
    synchro_transition_data_dictionary["m"] = m.tolist()
    synchro_transition_data_dictionary["W"] = W.tolist()
    synchro_transition_data_dictionary["K"] = K.tolist()
    synchro_transition_data_dictionary["A"] = A.tolist()
    synchro_transition_data_dictionary["reduced_W"] = reduced_W.tolist()
    synchro_transition_data_dictionary["reduced_K"] = reduced_K.tolist()
    synchro_transition_data_dictionary["reduced_A"] = reduced_A.tolist()
    synchro_transition_data_dictionary["n"] = n
    synchro_transition_data_dictionary["N"] = N
    synchro_transition_data_dictionary["theta0"] = theta0.tolist()
    synchro_transition_data_dictionary["T_1"] = T_1
    synchro_transition_data_dictionary["T_2"] = T_2
    synchro_transition_data_dictionary["T_3"] = T_3

    return synchro_transition_data_dictionary


def get_multiple_synchro_transition_kuramoto_graphs(t0, t1, dt, averaging,
                                                    CVM_dictionary, theta0,
                                                    sigma_array,
                                                    plot_time_series=False):
    """
    Generate data for the complete and the reduced Kuramoto dynamics
    on a particular graph. In particular, we get the global synchronization
    transition <R>_t vs sigma for the complete and the reduced dynamics for
    different target matrices.

    See synch_prediction/graphs/SBM/CVM_....py
                       or small_bipartite/CVM_....py
                       or two_triangles/CVM_....py


    :param t0: Initial time
    :param t1: Final time
    :param dt: Time step
    :param averaging: integer number X between 5 and 10. ex: 8
                       It means that the temporal series are averaged over
                       X*10% to 100% of their length
    :param CVM_dictionary: CVM_dictionary: it is the resulting dictionary
                           from the function:
                           get_CVM_dictionary .
                           See the documentation of the function.
    :param theta0: Initial conditions
    :param sigma_array: Array of the coupling constant sigma
    :param plot_time_series: Plot time series of the global synchro observable
                              vs time for the complete and the reduced dynamics

    :return: synchro_transition_data_dictionary : A dictionary that contains
                                                  the following keys,
      Keys
    { "r",                      -> Global synchro observables
                                   of the complete dynamics
      "R",                      -> Global synchro observables
                                   of the reduced dynamics
      # Not yet ..."r_mu_matrix",            -> Mesoscopic synchro observables
      # Not yet ...                             of the complete dynamics
      # Not yet ..."R_mu_matrix",            -> Mesoscopic synchro observables
      # Not yet ...                             of the reduced dynamics
      "sigma_array", "M", "l_normalized_weights", "m", "W", "K", "A",
      "reduced_W", "reduced_K", "reduced_A", "n", "N", "theta0", T_1, T_2, T_3}
    """

    multiple_synchro_transition_dictionary = {}

    W = np.array(CVM_dictionary["W"])
    K = np.array(CVM_dictionary["K"])
    A = np.array(CVM_dictionary["A"])

    N = len(A[0])
    n = len(np.array(CVM_dictionary["M_W"])[:, 0])
    targets_possibilities = ["W", "K", "A", "WK", "WA", "KW", "KA", "AW",
                             "AK", "WKA", "WAK", "KWA", "KAW", "AWK", "AKW"]

    for targets_string in targets_possibilities:

        if len(targets_string) == 1:
            T_1, T_2, T_3 = targets_string, "None", "None"
        elif len(targets_string) == 2:
            T_1, T_2 = list(targets_string)
            T_3 = "None"
        else:
            T_1, T_2, T_3 = list(targets_string)
        M = np.array(CVM_dictionary[f"M_{targets_string}"])
        m = global_reduction_matrix(M)

        # if reduction_matrix_has_rank_n(M) \
        #         and reduction_matrix_is_normalized(M):
        if matrix_is_positive(m) and matrix_is_normalized(m) \
                and matrix_is_positive(M) and matrix_is_normalized(M) \
                and matrix_has_rank_n(M):
            synchro_transition_data_dictionary = \
                get_synchro_transition_kuramoto_graphs(
                    t0, t1, dt, averaging, T_1, T_2, T_3, theta0, M,
                    sigma_array, W, A, plot_time_series=plot_time_series)
            multiple_synchro_transition_dictionary[
                "r_{}".format(targets_string)] \
                = synchro_transition_data_dictionary["r"]
            multiple_synchro_transition_dictionary[
                "R_{}".format(targets_string)] \
                = synchro_transition_data_dictionary["R"]
        else:
            multiple_synchro_transition_dictionary[
                "r_{}".format(targets_string)] \
                = {}
            multiple_synchro_transition_dictionary[
                "R_{}".format(targets_string)] \
                = {}
            # The empty dictionary means that the CVM setup (the reduction
            # matrix) for the choice of targets (targets_string)
            # was not giving a good observable.

    multiple_synchro_transition_dictionary["sigma_array"] = \
        sigma_array.tolist()
    multiple_synchro_transition_dictionary["W"] = W.tolist()
    multiple_synchro_transition_dictionary["K"] = K.tolist()
    multiple_synchro_transition_dictionary["A"] = A.tolist()
    multiple_synchro_transition_dictionary["n"] = n
    multiple_synchro_transition_dictionary["N"] = N
    multiple_synchro_transition_dictionary["theta0"] = theta0.tolist()

    return multiple_synchro_transition_dictionary


"""---------- Synchronization transition on random graphs ------------------"""
"""------------------- Old section III E. article --------------------------"""


# 1D reduction
def get_synchro_transition_theta(p_array, sizes, nb_instances, t0, t1, dt,
                                 Iext, sigma, N, plot_temporal_series=False):
    r_matrix = np.zeros((nb_instances, len(p_array)))
    R_matrix = np.zeros((nb_instances, len(p_array)))

    n = 0

    for p in tqdm(p_array):

        k = 0

        for instance in range(nb_instances):
            time.sleep(3)
            pin = p
            pout = p
            pq = [[pin, pout], [pout, pin]]
            A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
            m = get_eigenvectors_matrix(A, 1)
            theta0 = 2 * np.pi * np.random.rand(N)
            Z0 = np.sum(m * np.exp(1j * theta0))
            k_array = np.dot(A, np.ones(len(A[:, 0])).transpose())
            kappa = np.sum(m * k_array)
            hatkappa = np.sum(m**2*k_array) / np.sum(m ** 2)

            # Integrate complete dynamics
            args_complete_theta = (Iext, sigma)
            theta_sol = integrate_dynamics(t0, t1, dt, theta, A,
                                           "vode", theta0,
                                           *args_complete_theta)
            r = np.absolute(np.sum(m * np.exp(1j * theta_sol), axis=1))
            r_mean = np.sum(r[9 * int(t1 // dt) // 10:]
                            ) / len(r[9 * int(t1 // dt) // 10:])

            # Integrate recuced dynamics
            args_reduced_theta = (Iext, sigma, hatkappa, N)
            Z_sol = integrate_dynamics(t0, t1, dt, reduced_theta_1D, kappa,
                                       "zvode", Z0,
                                       *args_reduced_theta)
            R = np.absolute(Z_sol)
            R_mean = np.sum(R[9 * int(t1 // dt) // 10:]
                            ) / len(R[9 * int(t1 // dt) // 10:])

            r_matrix[k, n] = r_mean
            R_matrix[k, n] = R_mean

            if plot_temporal_series:

                # import matplotlib.pyplot as plt

                spike_array = (1 - np.cos(theta_sol)).transpose()
                for ind2 in range(0, int(t1 // dt)):
                    for ind1 in range(0, N):
                        if spike_array[ind1, ind2] < 1.5:
                            spike_array[ind1, ind2] = 0
                plt.matshow(spike_array, aspect="auto")
                plt.colorbar()
                plt.show()

                plt.plot(r, color="k")
                plt.plot(R, color="g", linestyle='--')
                plt.ylabel("$R$", fontsize=12)

                plt.show()

            k += 1

        n += 1

        # r_avg_instances_list.append(np.sum(r_list) / nb_instances)
        # R_avg_instances_list.append(np.sum(R_list) / nb_instances)

    return r_matrix, R_matrix


def get_synchro_transition_cowan_wilson(p_array, sizes, nb_instances,
                                        t0, t1, dt, tau, mu):
    N = np.sum(sizes)

    mean_x_instances_list = []
    X_instances_list = []
    var_x_instances_list = []
    varX_instances_list = []

    for p in tqdm(p_array):

        mean_x_list = []
        X_list = []
        var_x_list = []
        varX_list = []

        for instance in range(nb_instances):
            time.sleep(3)
            pin = p
            pout = p
            pq = [[pin, pout], [pout, pin]]
            A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
            V_not_normalized = get_eigenvectors_matrix(A, 1)
            V = V_not_normalized / np.sum(V_not_normalized[0])
            Vp = pinv(V)
            k_array = np.dot(A, np.ones(len(A[:, 0])).transpose())
            K = np.diag(k_array)
            kappa = np.dot(V, k_array)[0]
            hatkappa = multi_dot([V, K, Vp])[0][0]
            kappa2 = np.dot(V, k_array ** 2)[0]
            gamma = (multi_dot([(V * k_array), A, V.transpose()]
                               ) / np.dot(V, V.transpose()))[0][0]
            epsilon = kappa ** 2  # (multi_dot([(V**2), A,
            #  k_array.transpose()])/np.dot(V, V.transpose()))[0][0]
            tauR = (np.dot(V ** 2, np.diag(multi_dot([A, A, A]))) / np.sum(
                V ** 2 * k_array))[0]
            tauL = np.sum(
                (np.dot(V.transpose(), V) * A * np.dot(A, A))) / np.sum(
                V ** 2 * k_array)

            x0 = np.random.rand(N)
            X0 = np.dot(V[0], x0)
            varX0 = np.sum(V[0] * (x0 - X0) ** 2)
            Cxx0 = np.sum(V[0] * (x0 - X0) * np.dot(A, (x0 - X0)))
            W0 = np.array([X0, varX0, Cxx0])

            # Integrate complete dynamics
            args_cowan_wilson = (tau, mu)
            cowan_wilson_sol = integrate_dynamics(t0, t1, dt, cowan_wilson, A,
                                                  "vode", x0,
                                                  *args_cowan_wilson)
            mean_x = np.sum(V[0] * cowan_wilson_sol, axis=1)
            repeat_mean_x = np.tile(mean_x, (N, 1)).transpose()
            diff_xj_xmean = cowan_wilson_sol - repeat_mean_x
            var_x = np.sum(V * diff_xj_xmean ** 2, axis=1)

            mean_x_list.append(np.sum(mean_x[9 * int(t1 // dt) // 10:]) /
                               len(mean_x[9 * int(t1 // dt) // 10:]))
            var_x_list.append(np.sum(var_x[9 * int(t1 // dt) // 10:]) /
                              len(var_x[9 * int(t1 // dt) // 10:]))

            # Integrate recuced dynamics
            args_red_cowan_wilson = (tau, mu, hatkappa,
                                     kappa2, gamma, epsilon, tauL, tauR)
            red_cowan_wilson_sol = integrate_dynamics(t0, t1, dt,
                                                      reduced_cowan_wilson,
                                                      kappa, "vode", W0,
                                                      *args_red_cowan_wilson)
            Xt = red_cowan_wilson_sol[:, 0]
            varXt = red_cowan_wilson_sol[:, 1]
            X_list.append(Xt[9 * int(t1 // dt) // 10:] /
                          len(mean_x[9 * int(t1 // dt) // 10:]))
            varX_list.append(varXt[9 * int(t1 // dt) // 10:]
                             / len(varXt[9 * int(t1 // dt) // 10:]))

        # r_avg_instances_list.append(np.sum(r_list)/nb_instances)
        # R_avg_instances_list.append(np.sum(R_list)/nb_instances)

        mean_x_instances_list.append(np.sum(mean_x_list) / nb_instances)
        var_x_instances_list.append(np.sum(var_x_list) / nb_instances)
        X_instances_list.append(np.sum(X_list) / nb_instances)
        varX_instances_list.append(np.sum(varX_list) / nb_instances)

    return \
        mean_x_instances_list, var_x_instances_list, \
        X_instances_list, varX_instances_list


# 2D reduction
def get_synchro_transition_kuramoto_2D(p_out_array, p_in, sizes, nb_instances,
                                       t0, t1, dt, omega, omega_array,
                                       sigma, N, mean_SBM=False,
                                       plot_temporal_series=False):
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
                     stochastic block model, it is True. Warning: It allows
                     self-loops
    :param plot_temporal_series:

    :return: Matrices nb_instances x len(p_out_array) measuring the phase
             synchronization in each community and the global synchronization
    """

    n1, n2 = sizes

    r1_matrix = np.zeros((nb_instances, len(p_out_array)))
    r2_matrix = np.zeros((nb_instances, len(p_out_array)))
    R1_matrix = np.zeros((nb_instances, len(p_out_array)))
    R2_matrix = np.zeros((nb_instances, len(p_out_array)))
    r_matrix = np.zeros((nb_instances, len(p_out_array)))
    R_matrix = np.zeros((nb_instances, len(p_out_array)))

    m = 0

    for p_out in tqdm(p_out_array):

        k = 0

        for instance in range(nb_instances):
            time.sleep(3)

            pq = [[p_in, p_out], [p_out, p_in]]

            M_0 = np.block([[1/n1*np.ones(n1), np.zeros(n2)],
                            [np.zeros(n1), 1/n2*np.ones(n2)]])
            P = (np.block([[np.ones(n1), np.zeros(n2)],
                           [np.zeros(n1), np.ones(n2)]])).T
            
            if mean_SBM:
                print("Attention: la matrice moyenne definit ici "
                      "possede une diagonale"
                      "non nulle de sorte que "
                      "l'on attribue une probabilite non nulle"
                      "aux graphes du SBM d'avoir des boucles !!!")
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
                K = np.diag(np.sum(A, axis=0))

            else:
                A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
                K = np.diag(np.sum(A, axis=0))
                if p_in > 0:
                    M = M_0
                else:
                    V = get_eigenvectors_matrix(A, 2)  # Not normalized
                    Vp = pinv(V)
                    C = np.dot(M_0, Vp)
                    CV = np.dot(C, V)
                    M = (CV.T / (np.sum(CV, axis=1))).T

            alpha = np.sum(np.dot(M, A), axis=1)   # kappa in the article
            MAMp = multi_dot([M, A, pinv(M)])
            MKMp = multi_dot([M, K, pinv(M)])
            MWMp = multi_dot([M, np.diag(omega), pinv(M)])
            # Reduced paramters
            # redA = multi_dot([M, A, P])
            # hatredA = (multi_dot([M**2, A, P]).T / np.diag(np.dot(M, M.T))).T
            # hatLambda = multi_dot([M, A, pinv(M)])
            # print("\n", redA, "\n", "\n", hatredA, "\n", "\n", hatLambda)

            # Verifications

            # v_D = np.array([V[0, :]]).T
            # v_SD = np.array([V[1, :]]).T
            #
            # Lambda = np.linalg.eig(A)[0]
            # Lambda_D = (multi_dot([v_D.T, A, v_D]
            #                       )/np.dot(v_D.T, v_D))[0][0]
            # Lambda_SD = (multi_dot([v_SD.T, A, v_SD]
            #                        )/np.dot(v_SD.T, v_SD))[0][0]
            # Lambda_vec = np.array([Lambda_D, Lambda_SD])
            # Lambda_matrix = np.diag(Lambda_vec)
            #
            # print(C)
            # print(pinv(C))
            # print(np.dot(pinv(C), C))

            # print("\n Lambda =", Lambda, "\n",
            #       "\n Lambda = ", Lambda_matrix, "\n", "\n")
            #
            # print("\n hatLambda = ", hatLambda, "\n",
            #       "\n hatLambda = ", multi_dot([M, A, M.T]).T /
            #                                   (np.sum(M ** 2, axis=1)),
            #       "\n",
            #       "\n hatLambda = ", multi_dot([C, Lambda_matrix, pinv(C)]),
            #       "\n",
            #       "\n hatLambda = ", multi_dot([C, Lambda_matrix, C.T]
            #                                    ) / np.sum(C ** 2, axis=1),
            #        "\n",
            #       "\n hatLambda = ", np.dot(C, Lambda_vec)/np.sum(C, axis=1))
            #
            # print("\n", redA, "\n", hatredA, "\n", hatLambda)
            
            # import matplotlib.pyplot as plt
            # plt.plot()
            # plt.show()

            theta0 = 2*np.pi*np.random.rand(N)
            z0 = np.exp(1j * theta0)
            Z0 = np.dot(M, z0)

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
            r = np.absolute(
                np.sum((n1*M[0, :] + n2*M[1, :]) * np.exp(1j * kuramoto_sol),
                       axis=1))/N

            r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
            r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

            # Integrate recuced dynamics
            # args_red_kuramoto_2D = (omega_array, sigma, N,
            #                         hatredA, hatLambda)
            # red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
            #                                       reduced_kuramoto_2D,
            #                                       redA, "zvode", Z0,
            #                                       *args_red_kuramoto_2D)

            args_red_kuramoto_2D = (MWMp, sigma, N, MKMp, alpha)
            red_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                  reduced_kuramoto_2D2,
                                                  MAMp, "zvode", Z0,
                                                  *args_red_kuramoto_2D)
            Z1, Z2 = red_kuramoto_sol[:, 0], red_kuramoto_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1*Z1 + n2*Z2)/N

            R1_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                             ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                             ) / len(R2[5 * int(t1 // dt) // 10:])
            R_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                            ) / len(R[5 * int(t1 // dt) // 10:])

            r1_matrix[k, m] = r1_mean
            r2_matrix[k, m] = r2_mean
            R1_matrix[k, m] = R1_mean
            R2_matrix[k, m] = R2_mean
            r_matrix[k, m] = r_mean
            R_matrix[k, m] = R_mean

            if plot_temporal_series:

                import matplotlib.pyplot as plt

                plt.subplot(311)
                plt.plot(r, color="k")
                plt.plot(R, color="g")
                plt.plot(r_mean*np.ones(len(r)), color="r")
                plt.ylabel("$R$", fontsize=12)

                plt.subplot(312)
                plt.plot(r1, color="b")
                plt.plot(R1, color="lightblue")
                plt.ylabel("$R_1$", fontsize=12)

                plt.subplot(313)
                plt.plot(r2, color="r")
                plt.plot(R2, color="y")
                plt.ylabel("$R_2$", fontsize=12)
                plt.xlabel("$t$", fontsize=12)
                plt.tight_layout()

                plt.show()

            k += 1

        m += 1

    return r1_matrix, r2_matrix, R1_matrix, R2_matrix, r_matrix, R_matrix


def get_synchro_transition_theta_2D(p_out_array, p_in, sizes, nb_instances,
                                    t0, t1, dt, I_ext, I_array,
                                    sigma, N, mean_SBM=False,
                                    plot_temporal_series=False):
    """
    Predictions for the theta dynamics on modular graphs when one module has
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
    :param I_ext: 1xN : n1*[I1] + n2*[I2]
    :param I_array: 1x2 : [I1, I2]
    :param sigma: Coupling constant
    :param N: Number of nodes
    :param mean_SBM: (Boolean)If we want to get the transition for the mean
                     stochastic block model, it is True. Warning: It allows
                     self loops !!!
    :param plot_temporal_series:

    :return: Matrices nb_instances x len(p_out_array) measuring the phase
             synchronization in each community and the global synchronization
    """

    n1, n2 = sizes

    r1_matrix = np.zeros((nb_instances, len(p_out_array)))
    r2_matrix = np.zeros((nb_instances, len(p_out_array)))
    R1_matrix = np.zeros((nb_instances, len(p_out_array)))
    R2_matrix = np.zeros((nb_instances, len(p_out_array)))
    r_matrix = np.zeros((nb_instances, len(p_out_array)))
    R_matrix = np.zeros((nb_instances, len(p_out_array)))

    m = 0

    for p_out in tqdm(p_out_array):

        k = 0

        for instance in range(nb_instances):
            time.sleep(3)

            pq = [[p_in, p_out], [p_out, p_in]]

            M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                            [np.zeros(n1), 1 / n2 * np.ones(n2)]])
            P = (np.block([[np.ones(n1), np.zeros(n2)],
                           [np.zeros(n1), np.ones(n2)]])).T

            if mean_SBM:
                print("Attention: la matrice moyenne definit ici "
                      "possede une diagonale"
                      "non nulle de sorte que "
                      "l'on attribue une probabilite non nulle"
                      "aux graphes du SBM d'avoir des boucles !!!")
                # A = np.zeros((N, N))
                # ii = 0
                # for i in range(0, len(sizes)):
                #     jj = 0
                #     for j in range(0, len(sizes)):
                #         A[ii:ii + sizes[i], jj:jj + sizes[j]] \
                #             = pq[i][j] * np.ones((sizes[i], sizes[j]))
                #         jj += sizes[j]
                #     ii += sizes[i]
                A = np.block([[pq[0][0]*np.ones((n2, n1)),
                               pq[0][1]*np.ones((n1, n2))],
                              [pq[1][0]*np.ones((n2, n1)),
                               pq[1][1]*np.ones((n2, n1))]])
                M = M_0

            else:
                A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
                if p_in > 0:
                    M = M_0
                else:
                    V = get_eigenvectors_matrix(A, 2)  # Not normalized
                    Vp = pinv(V)
                    C = np.dot(M_0, Vp)
                    CV = np.dot(C, V)
                    M = (CV.T / (np.sum(CV, axis=1))).T

            # Reduced paramters
            redA = multi_dot([M, A, P])
            hatredA = (multi_dot([M ** 2, A, P]).T / np.diag(np.dot(M, M.T))).T
            hatLambda = multi_dot([M, A, pinv(M)])

            theta0 = 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            Z0 = np.dot(M, z0)

            # Integrate complete dynamics
            args_theta = (I_ext, sigma)
            theta_sol = integrate_dynamics(t0, t1, dt, theta, A,
                                           "vode", theta0,
                                           *args_theta)

            r1 = np.absolute(
                np.sum(M[0, 0:n1] * np.exp(1j * theta_sol[:, 0:n1]),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, n1:] * np.exp(1j * theta_sol[:, n1:]),
                       axis=1))
            r = np.absolute(
                np.sum((n1*M[0, :] + n2*M[1, :]) * np.exp(1j * theta_sol),
                       axis=1))/N

            r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
            r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

            # Integrate recuced dynamics
            args_red_theta_2D = (I_array, sigma, N,
                                 hatredA, hatLambda)
            red_theta_sol = integrate_dynamics(t0, t1, dt,
                                               reduced_theta_2D,
                                               redA, "zvode", Z0,
                                               *args_red_theta_2D)

            Z1, Z2 = red_theta_sol[:, 0], red_theta_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1*Z1 + n2*Z2)/N

            R1_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                             ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                             ) / len(R2[5 * int(t1 // dt) // 10:])
            R_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                            ) / len(R[5 * int(t1 // dt) // 10:])

            r1_matrix[k, m] = r1_mean
            r2_matrix[k, m] = r2_mean
            R1_matrix[k, m] = R1_mean
            R2_matrix[k, m] = R2_mean
            r_matrix[k, m] = r_mean
            R_matrix[k, m] = R_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.subplot(411)
                plt.plot(r, color="k")
                plt.plot(R, color="g")
                plt.ylabel("$R$", fontsize=12)
                plt.ylim([0, 1.02])

                plt.subplot(412)
                plt.plot(r1, color="b")
                plt.plot(R1, color="lightblue")
                plt.ylabel("$R_1$", fontsize=12)
                plt.ylim([0, 1.02])

                plt.subplot(413)
                plt.plot(r2, color="r")
                plt.plot(R2, color="y")
                plt.ylabel("$R_2$", fontsize=12)
                plt.xlabel("$t$", fontsize=12)
                plt.ylim([0, 1.02])

                plt.subplot(414)
                plt.plot()
                plt.tight_layout()

                plt.show()

            k += 1

        m += 1

    return r1_matrix, r2_matrix, R1_matrix, R2_matrix, r_matrix, R_matrix


def get_synchro_transition_winfree_2D(p_out_array, p_in, sizes, nb_instances,
                                      t0, t1, dt, omega, omega_array,
                                      sigma, N, mean_SBM=False,
                                      plot_temporal_series=False):
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

    r1_matrix = np.zeros((nb_instances, len(p_out_array)))
    r2_matrix = np.zeros((nb_instances, len(p_out_array)))
    R1_matrix = np.zeros((nb_instances, len(p_out_array)))
    R2_matrix = np.zeros((nb_instances, len(p_out_array)))
    r_matrix = np.zeros((nb_instances, len(p_out_array)))
    R_matrix = np.zeros((nb_instances, len(p_out_array)))

    rg_matrix = np.zeros((nb_instances, len(p_out_array)))

    m = 0

    for p_out in tqdm(p_out_array):

        k = 0

        for instance in range(nb_instances):
            time.sleep(3)

            pq = [[p_in, p_out], [p_out, p_in]]

            M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                            [np.zeros(n1), 1 / n2 * np.ones(n2)]])
            P = (np.block([[np.ones(n1), np.zeros(n2)],
                           [np.zeros(n1), np.ones(n2)]])).T

            if mean_SBM:
                print("Attention: la matrice moyenne definit ici "
                      "possede une diagonale"
                      "non nulle de sorte que "
                      "l'on attribue une probabilite non nulle"
                      "aux graphes du SBM d'avoir des boucles !!!")
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
                K = np.diag(np.sum(A, axis=0))

            else:
                A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
                K = np.diag(np.sum(A, axis=0))
                if p_in > 0:
                    M = M_0
                else:
                    V = get_eigenvectors_matrix(A, 2)  # Not normalized
                    Vp = pinv(V)
                    C = np.dot(M_0, Vp)
                    CV = np.dot(C, V)
                    M = (CV.T / (np.sum(CV, axis=1))).T

            alpha = np.sum(np.dot(M, A), axis=1)   # kappa in the article
            MAMp = multi_dot([M, A, pinv(M)])
            MKMp = multi_dot([M, K, pinv(M)])
            MWMp = multi_dot([M, np.diag(omega), pinv(M)])

            # Reduced paramters
            # redA = multi_dot([M, A, P])
            # hatredA = (multi_dot([M ** 2, A, P]).T / np.diag(np.dot(M, M.T))).T
            # hatLambda = multi_dot([M, A, pinv(M)])

            # print(redA, hatredA, hatLambda)

            # Verifications

            # v_D = np.array([V[0, :]]).T
            # v_SD = np.array([V[1, :]]).T
            #
            # Lambda = np.linalg.eig(A)[0]
            # Lambda_D = (multi_dot([v_D.T, A, v_D]
            #                       )/np.dot(v_D.T, v_D))[0][0]
            # Lambda_SD = (multi_dot([v_SD.T, A, v_SD]
            #                        )/np.dot(v_SD.T, v_SD))[0][0]
            # Lambda_vec = np.array([Lambda_D, Lambda_SD])
            # Lambda_matrix = np.diag(Lambda_vec)
            #
            # print(C)
            # print(pinv(C))
            # print(np.dot(pinv(C), C))

            # print("\n Lambda =", Lambda, "\n",
            #       "\n Lambda = ", Lambda_matrix, "\n", "\n")
            #
            # print("\n hatLambda = ", hatLambda, "\n",
            #       "\n hatLambda = ", multi_dot([M, A, M.T]).T /
            #                                   (np.sum(M ** 2, axis=1)),
            #       "\n",
            #       "\n hatLambda = ", multi_dot([C, Lambda_matrix, pinv(C)]),
            #       "\n",
            #       "\n hatLambda = ", multi_dot([C, Lambda_matrix, C.T]
            #                                    ) / np.sum(C ** 2, axis=1),
            #        "\n",
            #       "\n hatLambda = ", np.dot(C, Lambda_vec)/np.sum(C, axis=1))
            #
            # print("\n", redA, "\n", hatredA, "\n", hatLambda)

            # import matplotlib.pyplot as plt
            # plt.plot()
            # plt.show()

            theta0 = 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            Z0 = np.dot(M, z0)

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
            r = np.absolute(
                np.sum((n1*M[0, :] + n2*M[1, :]) * np.exp(1j * winfree_sol),
                       axis=1))/N
            rg = np.absolute(np.sum(np.exp(1j * winfree_sol), axis=1))/N

            r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
            r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])
            rg_mean = np.sum(rg[5 * int(t1 // dt) // 10:]
                             ) / len(rg[5 * int(t1 // dt) // 10:])

            # Integrate recuced dynamics
            # args_red_winfree_2D = (omega_array, sigma, N,
            #                        hatredA, hatLambda)
            # red_winfree_sol = integrate_dynamics(t0, t1, dt,
            #                                      reduced_winfree_2D,
            #                                      redA, "zvode", Z0,
            #                                      *args_red_winfree_2D)
            args_red_winfree_2D = (MWMp, sigma, N, MKMp, alpha)
            red_winfree_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_winfree_2D2,
                                                 MAMp, "zvode", Z0,
                                                 *args_red_winfree_2D)

            Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1*Z1 + n2*Z2)/N

            R1_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                             ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                             ) / len(R2[5 * int(t1 // dt) // 10:])
            R_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                            ) / len(R[5 * int(t1 // dt) // 10:])

            r1_matrix[k, m] = r1_mean
            r2_matrix[k, m] = r2_mean
            R1_matrix[k, m] = R1_mean
            R2_matrix[k, m] = R2_mean
            r_matrix[k, m] = r_mean
            R_matrix[k, m] = R_mean
            rg_matrix[k, m] = rg_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.subplot(311)
                plt.plot(r, color="k")
                plt.plot(R, color="g")
                plt.plot(r_mean*np.ones(len(r)), color="r")
                plt.ylabel("$R$", fontsize=12)
                plt.ylim([-0.02, 1.02])

                plt.subplot(312)
                plt.plot(r1, color="b")
                plt.plot(R1, color="lightblue")
                plt.ylabel("$R_1$", fontsize=12)
                plt.ylim([-0.02, 1.02])

                plt.subplot(313)
                plt.plot(r2, color="r")
                plt.plot(R2, color="y")
                plt.ylabel("$R_2$", fontsize=12)
                plt.ylim([-0.02, 1.02])
                plt.xlabel("$t$", fontsize=12)
                plt.tight_layout()

                plt.show()

            k += 1

        m += 1

    return \
        r1_matrix, r2_matrix, R1_matrix, R2_matrix,\
        r_matrix, R_matrix, rg_matrix


def get_synchro_transition_cowan_wilson_2D(p_out_array, p_11, p_22, sizes,
                                           nb_instances, t0, t1, dt, tau, mu):
    # TODO: La somme n'est pas la bonne ici pour la dynamique complète!
    # On devrait sommer sur les module B_mu à la place !

    n1, n2 = sizes

    N = np.sum(sizes)

    mean_x1_instances_list, mean_x2_instances_list = [], []
    X1_instances_list, X2_instances_list = [], []
    # var_x1_instances_list, var_x2_instances_list = [], []
    # varX1_instances_list, varX2_instances_list = [], []
    r1_instances_list, r2_instances_list = [], []
    R1_instances_list, R2_instances_list = [], []

    for p_out in tqdm(p_out_array):

        mean_x1_list, mean_x2_list = [], []
        X1_list, X2_list = [], []
        # var_x1_list, var_x2_list = [], []
        # varX1_list, varX2_list = [], []
        r1_list, r2_list = [], []
        R1_list, R2_list = [], []

        for instance in range(nb_instances):
            time.sleep(3)
            pq = [[p_11, p_out], [p_out, p_22]]
            A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
            # A = np.zeros((N, N))
            # print("Attention: la matrice moyenne definit ici "
            #       "possede une diagonale"
            #       "non nulle de sorte que "
            #       "l'on attribue une probabilite non nulle"
            #       "aux graphes du SBM d'avoir des boucles !!!")
            # ii = 0
            # for i in range(0, len(sizes)):
            #     jj = 0
            #     for j in range(0, len(sizes)):
            #         A[ii:ii + sizes[i], jj:jj + sizes[j]] \
            #             = pq[i][j]*np.ones((sizes[i], sizes[j]))
            #         jj += sizes[j]
            #     ii += sizes[i]
            V = get_eigenvectors_matrix(A, 2)  # Not normalized
            # plt.matshow(A, aspect="auto")
            # plt.show()
            M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                            [np.zeros(n1), 1 / n2 * np.ones(n2)]])
            P = (np.block([[np.ones(n1), np.zeros(n2)],
                           [np.zeros(n1), np.ones(n2)]])).T
            Vp = pinv(V)
            C = np.dot(M_0, Vp)
            CV = np.dot(C, V)
            M = (CV.T / (np.sum(CV, axis=1))).T
            # M = M_0
            # M_min_max_norm = ((CV.T - np.min(CV, axis=1)) / (
            #         np.max(CV, axis=1) - np.min(CV, axis=1))).T
            # M = (M_min_max_norm.T / (np.sum(M_min_max_norm, axis=1))).T

            # Reduced paramters
            redA = multi_dot([M, A, P])
            redK = (multi_dot([M ** 2, A, P]).T / np.diag(np.dot(M, M.T))).T
            hatLambda = multi_dot([M, A, pinv(M)])

            x0 = np.random.rand(N)
            X0 = np.dot(M, x0)
            varX0 = np.dot(M, x0 ** 2) - X0 ** 2
            # Cxx0 = np.sum(V*(x0 - X0)*np.dot(A, (x0 - X0)))
            W0 = np.concatenate([X0, varX0])  # , Cxx0])

            # Integrate complete dynamics
            args_cowan_wilson = (tau, mu)
            cowan_wilson_sol = integrate_dynamics(t0, t1, dt, cowan_wilson, A,
                                                  "vode", x0,
                                                  *args_cowan_wilson)

            # TODO: La somme n'est pas la bonne ici ! On devrait sommer sur les
            # module B_mu à la place !

            mean_x = np.dot(M, cowan_wilson_sol.T).T
            var_x = np.dot(M, cowan_wilson_sol.T ** 2).T - mean_x ** 2
            mean_x1, mean_x2 = mean_x[:, 0], mean_x[:, 1]
            # var_x1, var_x2 = var_x[:, 0], var_x[:, 1]
            r1 = np.exp(-np.sqrt(np.abs(var_x[:, 0]) / mean_x1 ** 2))
            r2 = np.exp(-np.sqrt(np.abs(var_x[:, 1]) / mean_x2 ** 2))

            mean_x1_list.append(np.sum(mean_x1[9 * int(t1 // dt) // 10:]) /
                                len(mean_x1[9 * int(t1 // dt) // 10:]))
            mean_x2_list.append(np.sum(mean_x2[9 * int(t1 // dt) // 10:]) /
                                len(mean_x2[9 * int(t1 // dt) // 10:]))
            # var_x1_list.append(np.sum(var_x1[9*int(t1//dt)//10:]) /
            #                   len(var_x1[9*int(t1//dt)//10:]))
            # var_x2_list.append(np.sum(var_x2[9 * int(t1 // dt) // 10:]) /
            #                    len(var_x2[9 * int(t1 // dt) // 10:]))
            r1_list.append(np.sum(r1[9 * int(t1 // dt) // 10:]) /
                           len(r1[9 * int(t1 // dt) // 10:]))
            r2_list.append(np.sum(r2[9 * int(t1 // dt) // 10:]) /
                           len(r2[9 * int(t1 // dt) // 10:]))

            # Integrate recuced dynamics
            args_red_cowan_wilson_2D = (tau, mu, hatLambda, redK)
            red_cowan_wilson_sol = \
                integrate_dynamics(t0, t1, dt, reduced_cowan_wilson_2D,
                                   redA, "vode", W0, *args_red_cowan_wilson_2D)

            sol = red_cowan_wilson_sol
            X1, X2 = sol[:, 0], sol[:, 1]
            # varX1, varX2 = sol[:, 2], sol[:, 3]

            R1 = np.exp(-np.sqrt(np.abs(sol[:, 2]) / X1 ** 2))
            R2 = np.exp(-np.sqrt(np.abs(sol[:, 3]) / X2 ** 2))

            # print(r1, r2)
            # print(R1, R2)

            # The absolute value in R_mu is used because we can't guarentee the
            # positiveness of the variance in the reduced dynamics.

            X1_list.append(X1[9 * int(t1 // dt) // 10:] /
                           len(X1[9 * int(t1 // dt) // 10:]))
            X2_list.append(X2[9 * int(t1 // dt) // 10:] /
                           len(X2[9 * int(t1 // dt) // 10:]))
            # varX1_list.append(varX1[9*int(t1//dt)//10:] /
            #                   len(varX1[9*int(t1//dt)//10:]))
            # varX2_list.append(varX2[9 * int(t1 // dt) // 10:]
            #                   / len(varX2[9 * int(t1 // dt) // 10:]))
            R1_list.append(R1[9 * int(t1 // dt) // 10:] /
                           len(R1[9 * int(t1 // dt) // 10:]))
            R2_list.append(R2[9 * int(t1 // dt) // 10:] /
                           len(R2[9 * int(t1 // dt) // 10:]))

        # r_avg_instances_list.append(np.sum(r_list)/nb_instances)
        # R_avg_instances_list.append(np.sum(R_list)/nb_instances)

        mean_x1_instances_list.append(np.sum(mean_x1_list) / nb_instances)
        mean_x2_instances_list.append(np.sum(mean_x2_list) / nb_instances)
        # var_x1_instances_list.append(np.sum(var_x1_list)/nb_instances)
        # var_x2_instances_list.append(np.sum(var_x2_list)/nb_instances)
        r1_instances_list.append(np.sum(r1_list) / nb_instances)
        r2_instances_list.append(np.sum(r2_list) / nb_instances)

        X1_instances_list.append(np.sum(X1_list) / nb_instances)
        X2_instances_list.append(np.sum(X2_list) / nb_instances)
        # varX1_instances_list.append(np.sum(varX1_list)/nb_instances)
        # varX2_instances_list.append(np.sum(varX2_list)/nb_instances)
        R1_instances_list.append(np.sum(R1_list) / nb_instances)
        R2_instances_list.append(np.sum(R2_list) / nb_instances)

    return \
        mean_x1_instances_list, mean_x2_instances_list, r1_instances_list, \
        r2_instances_list, X1_instances_list, X2_instances_list, \
        R1_instances_list, R2_instances_list


def get_synchro_transition_cosinus_2D(p_out_array, p_in, sizes, nb_instances,
                                      t0, t1, dt, omega, omega_array,
                                      sigma, N, mean_SBM=False,
                                      plot_temporal_series=False):
    """
    Predictions for the cosinus dynamics on modular graphs when one module has
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

    r1_matrix = np.zeros((nb_instances, len(p_out_array)))
    r2_matrix = np.zeros((nb_instances, len(p_out_array)))
    R1_matrix = np.zeros((nb_instances, len(p_out_array)))
    R2_matrix = np.zeros((nb_instances, len(p_out_array)))
    r_matrix = np.zeros((nb_instances, len(p_out_array)))
    R_matrix = np.zeros((nb_instances, len(p_out_array)))

    m = 0

    for p_out in tqdm(p_out_array):

        k = 0

        for instance in range(nb_instances):
            time.sleep(3)

            pq = [[p_in, p_out], [p_out, p_in]]

            M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
                            [np.zeros(n1), 1 / n2 * np.ones(n2)]])
            P = (np.block([[np.ones(n1), np.zeros(n2)],
                           [np.zeros(n1), np.ones(n2)]])).T

            if mean_SBM:
                print("Attention: la matrice moyenne definit ici "
                      "possede une diagonale"
                      "non nulle de sorte que "
                      "l'on attribue une probabilite non nulle"
                      "aux graphes du SBM d'avoir des boucles !!!")
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
                    V = get_eigenvectors_matrix(A, 2)  # Not normalized
                    Vp = pinv(V)
                    C = np.dot(M_0, Vp)
                    CV = np.dot(C, V)
                    M = (CV.T / (np.sum(CV, axis=1))).T

            # Reduced paramters
            redA = multi_dot([M, A, P])
            hatredA = (multi_dot([M ** 2, A, P]).T / np.diag(np.dot(M, M.T))).T
            hatLambda = multi_dot([M, A, pinv(M)])

            # print("\n", redA, "\n", "\n", hatredA, "\n", "\n", hatLambda)

            theta0 = np.linspace(0, 2*np.pi, N)   # 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            Z0 = np.dot(M, z0)

            # Integrate complete dynamics
            args_cosinus = (omega, sigma)
            cosinus_sol = integrate_dynamics(t0, t1, dt, cosinus, A,
                                             "vode", theta0,
                                             *args_cosinus)

            r1 = np.absolute(
                np.sum(M[0, 0:n1] * np.exp(1j * cosinus_sol[:, 0:n1]),
                       axis=1))
            r2 = np.absolute(
                np.sum(M[1, n1:] * np.exp(1j * cosinus_sol[:, n1:]),
                       axis=1))
            r = np.absolute(
                np.sum(
                    (n1 * M[0, :] + n2 * M[1, :]) * np.exp(1j * cosinus_sol),
                    axis=1)) / N

            r1_mean = np.sum(r1[5 * int(t1 // dt) // 10:]
                             ) / len(r1[5 * int(t1 // dt) // 10:])
            r2_mean = np.sum(r2[5 * int(t1 // dt) // 10:]
                             ) / len(r2[5 * int(t1 // dt) // 10:])
            r_mean = np.sum(r[5 * int(t1 // dt) // 10:]
                            ) / len(r[5 * int(t1 // dt) // 10:])

            # Integrate recuced dynamics
            args_red_cosinus_2D = (omega_array, sigma, N,
                                   hatredA, hatLambda)
            red_cosinus_sol = integrate_dynamics(t0, t1, dt,
                                                 reduced_cosinus_2D,
                                                 redA, "zvode", Z0,
                                                 *args_red_cosinus_2D)

            Z1, Z2 = red_cosinus_sol[:, 0], red_cosinus_sol[:, 1]
            R1, R2 = np.absolute(Z1), np.absolute(Z2)
            R = np.absolute(n1 * Z1 + n2 * Z2) / N

            R1_mean = np.sum(R1[5 * int(t1 // dt) // 10:]
                             ) / len(R1[5 * int(t1 // dt) // 10:])
            R2_mean = np.sum(R2[5 * int(t1 // dt) // 10:]
                             ) / len(R2[5 * int(t1 // dt) // 10:])
            R_mean = np.sum(R[5 * int(t1 // dt) // 10:]
                            ) / len(R[5 * int(t1 // dt) // 10:])

            r1_matrix[k, m] = r1_mean
            r2_matrix[k, m] = r2_mean
            R1_matrix[k, m] = R1_mean
            R2_matrix[k, m] = R2_mean
            r_matrix[k, m] = r_mean
            R_matrix[k, m] = R_mean

            if plot_temporal_series:
                import matplotlib.pyplot as plt

                plt.subplot(311)
                plt.plot(r, color="k")
                plt.plot(R, color="g")
                # plt.plot(r_mean * np.ones(len(r)), color="r")
                plt.ylabel("$R$", fontsize=12)
                plt.ylim([0, 1.02])

                plt.subplot(312)
                plt.plot(r1, color="b")
                plt.plot(R1, color="lightblue")
                plt.ylabel("$R_1$", fontsize=12)
                plt.ylim([0, 1.02])

                plt.subplot(313)
                plt.plot(r2, color="r")
                plt.plot(R2, color="y")
                plt.ylabel("$R_2$", fontsize=12)
                plt.xlabel("$t$", fontsize=12)
                plt.tight_layout()
                plt.ylim([0, 1.02])

                plt.show()

            k += 1

        m += 1

    return r1_matrix, r2_matrix, R1_matrix, R2_matrix, r_matrix, R_matrix


# def get_synchro_transition_winfree_2D(p_out_array, p_in, sizes, nb_instances,
#                                       t0, t1, dt, omega, omega_array,
#                                       sigma, N, mean_SBM=False,
#                                       plot_temporal_series=False):
#    """
#    Predictions for the Winfree dynamics on modular graphs when one module has
#    one natural frequency and the other has another natural frequency
#
#    :param p_out_array: Array of probabilities, for a node, to be connected
#                        outside its community.
#   :param p_in: Probability, for a node, to be connected inside its community.
#                 If p_in = 0, it is a SBM graph.
#    :param sizes: [n1, n2] Community sizes
#    :param nb_instances: The number of instances of graphs
#                         and initial conditions
#    :param t0: Initial time
#    :param t1: Final time
#    :param dt: Time step
#    :param omega: 1xN : n1*[omega1] + n2*[omega2]
#    :param omega_array: 1x2 : [omega1, omega2]
#    :param sigma: Coupling constant
#    :param N: Number of nodes
#    :param mean_SBM: (Boolean)If we want to get the transition for the mean
#                     stochastic block model, it is True.
#    :param plot_temporal_series
#
#    :return: Matrices nb_instances x len(p_out_array) measuring the phase
#             synchronization in each community and the global synchronization
#    """
#
#    n1, n2 = sizes
#
#    r1_matrix = np.zeros((nb_instances, len(p_out_array)))
#    phi1_matrix = np.zeros((nb_instances, len(p_out_array)))
#    r2_matrix = np.zeros((nb_instances, len(p_out_array)))
#    phi2_matrix = np.zeros((nb_instances, len(p_out_array)))
#    R1_matrix = np.zeros((nb_instances, len(p_out_array)))
#    Phi1_matrix = np.zeros((nb_instances, len(p_out_array)))
#    R2_matrix = np.zeros((nb_instances, len(p_out_array)))
#    Phi2_matrix = np.zeros((nb_instances, len(p_out_array)))
#    r_matrix = np.zeros((nb_instances, len(p_out_array)))
#    phi_matrix = np.zeros((nb_instances, len(p_out_array)))
#    R_matrix = np.zeros((nb_instances, len(p_out_array)))
#    Phi_matrix = np.zeros((nb_instances, len(p_out_array)))
#
#    m = 0
#
#    for p_out in tqdm(p_out_array):
#
#        k = 0
#
#        for instance in range(nb_instances):
#            time.sleep(3)
#
#            pq = [[p_in, p_out], [p_out, p_in]]
#
#            M_0 = np.block([[1 / n1 * np.ones(n1), np.zeros(n2)],
#                            [np.zeros(n1), 1 / n2 * np.ones(n2)]])
#            P = (np.block([[np.ones(n1), np.zeros(n2)],
#                           [np.zeros(n1), np.ones(n2)]])).T
#
#            if mean_SBM:
#                A = np.zeros((N, N))
#                ii = 0
#                for i in range(0, len(sizes)):
#                    jj = 0
#                    for j in range(0, len(sizes)):
#                        A[ii:ii + sizes[i], jj:jj + sizes[j]] \
#                            = pq[i][j] * np.ones((sizes[i], sizes[j]))
#                        jj += sizes[j]
#                    ii += sizes[i]
#
#                M = M_0
#
#            else:
#                A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
#                if p_in > 0:
#                    M = M_0
#                else:
#                    V = get_eigenvectors_matrix(A, 2)  # Not normalized
#                    Vp = pinv(V)
#                    C = np.dot(M_0, Vp)
#                    CV = np.dot(C, V)
#                    M = (CV.T / (np.sum(CV, axis=1))).T
#
#            # Reduced paramters
#            redA = multi_dot([M, A, P])
#            hatredA = (multi_dot([M**2, A, P]).T / np.diag(np.dot(M, M.T))).T
#            hatLambda = multi_dot([M, A, pinv(M)])
#
#            # print(redA, "\n", "\n", hatredA, "\n", "\n", hatLambda)
#
#            # Verifications
#
#            # v_D = np.array([V[0, :]]).T
#            # v_SD = np.array([V[1, :]]).T
#            #
#            # Lambda = np.linalg.eig(A)[0]
#            # Lambda_D = (multi_dot([v_D.T, A, v_D]
#            #                       )/np.dot(v_D.T, v_D))[0][0]
#            # Lambda_SD = (multi_dot([v_SD.T, A, v_SD]
#            #                        )/np.dot(v_SD.T, v_SD))[0][0]
#            # Lambda_vec = np.array([Lambda_D, Lambda_SD])
#            # Lambda_matrix = np.diag(Lambda_vec)
#            #
#            # print(C)
#            # print(pinv(C))
#            # print(np.dot(pinv(C), C))
#
#            # print("\n Lambda =", Lambda, "\n",
#            #       "\n Lambda = ", Lambda_matrix, "\n", "\n")
#            #
#            # print("\n hatLambda = ", hatLambda, "\n",
#            #       "\n hatLambda = ", multi_dot([M, A, M.T]).T /
#            #                                   (np.sum(M ** 2, axis=1)),
#            #       "\n",
#            #       "\n hatLambda = ", multi_dot([C, Lambda_matrix, pinv(C)]),
#            #       "\n",
#            #       "\n hatLambda = ", multi_dot([C, Lambda_matrix, C.T]
#            #                                    ) / np.sum(C ** 2, axis=1),
#            #        "\n",
#            #      "\n hatLambda = ", np.dot(C, Lambda_vec)/np.sum(C, axis=1))
#            #
#            # print("\n", redA, "\n", hatredA, "\n", hatLambda)
#            #
#            # import matplotlib.pyplot as plt
#            # plt.matshow(pinv(M), aspect="auto")
#            # plt.colorbar()
#            # plt.show()
#
#            theta0 = 2 * np.pi * np.random.rand(N)
#            z0 = np.exp(1j * theta0)
#            Z0 = np.dot(M, z0)
#
#            # Integrate complete dynamics
#            args_winfree = (omega, sigma)
#            winfree_sol = integrate_dynamics(t0, t1, dt, winfree, A,
#                                             "vode", theta0,
#                                             *args_winfree)
#            z1 = np.sum(M[0, 0:n1]*np.exp(1j * winfree_sol[:, 0:n1]), axis=1)
#            z2 = np.sum(M[1, n1:]*np.exp(1j * winfree_sol[:, n1:]), axis=1)
#            z = 0.5*(z1 + z2)
#
#            z1_mean = np.sum(z1[5 * int(t1 // dt) // 10:]
#                             ) / len(z1[5 * int(t1 // dt) // 10:])
#            z2_mean = np.sum(z2[5 * int(t1 // dt) // 10:]
#                             ) / len(z2[5 * int(t1 // dt) // 10:])
#            z_mean = np.sum(z[5 * int(t1 // dt) // 10:]
#                            ) / len(z[5 * int(t1 // dt) // 10:])
#
#            r1_mean, phi1_mean = np.absolute(z1_mean), np.angle(z1_mean)
#            r2_mean, phi2_mean = np.absolute(z2_mean), np.angle(z2_mean)
#            r_mean, phi_mean = np.absolute(z_mean), np.angle(z_mean)
#
#            print(r_mean)
#
#            # Integrate recuced dynamics
#            args_red_winfree_2D = (omega_array, sigma, N,
#                                   hatredA, hatLambda)
#            red_winfree_sol = integrate_dynamics(t0, t1, dt,
#                                                 reduced_winfree_2D,
#                                                 redA, "zvode", Z0,
#                                                 *args_red_winfree_2D)
#
#            Z1, Z2 = red_winfree_sol[:, 0], red_winfree_sol[:, 1]
#            Z = 0.5*(Z1 + Z2)
#
#            Z1_mean = np.sum(Z1[5 * int(t1 // dt) // 10:]
#                             ) / len(Z1[5 * int(t1 // dt) // 10:])
#            Z2_mean = np.sum(Z2[5 * int(t1 // dt) // 10:]
#                             ) / len(Z2[5 * int(t1 // dt) // 10:])
#            Z_mean = np.sum(Z[5 * int(t1 // dt) // 10:]
#                            ) / len(Z[5 * int(t1 // dt) // 10:])
#
#            R1_mean, Phi1_mean = np.absolute(Z1_mean), np.angle(Z1_mean)
#            R2_mean, Phi2_mean = np.absolute(Z2_mean), np.angle(Z2_mean)
#            R_mean, Phi_mean = np.absolute(Z_mean), np.angle(Z_mean)
#
#            r1_matrix[k, m], phi1_matrix[k, m] = r1_mean, phi1_mean
#            r2_matrix[k, m], phi2_matrix[k, m] = r2_mean, phi2_mean
#            R1_matrix[k, m], Phi1_matrix[k, m] = R1_mean, Phi1_mean
#            R2_matrix[k, m], Phi2_matrix[k, m] = R2_mean, Phi2_mean
#            r_matrix[k, m], phi_matrix[k, m] = r_mean, phi_mean
#            R_matrix[k, m], Phi_matrix[k, m] = R_mean, Phi_mean
#
#            # See the temporal series
#            if plot_temporal_series:
#                import matplotlib.pyplot as plt
#
#                plt.subplot(311)
#                plt.plot(np.absolute(z_mean)*np.ones(len(z)))
#                plt.plot(np.absolute(z), color="k")
#                plt.plot(np.absolute(Z), color="g")
#                plt.ylabel("$R$", fontsize=12)
#                plt.ylim([-0.02, 1.02])
#
#                plt.subplot(312)
#                plt.plot(np.absolute(z1), color="b")
#                plt.plot(np.absolute(Z1), color="lightblue")
#                plt.ylabel("$R_1$", fontsize=12)
#                plt.ylim([-0.02, 1.02])
#
#                plt.subplot(313)
#                plt.plot(np.absolute(z2), color="r")
#                plt.plot(np.absolute(Z2), color="y")
#                plt.ylabel("$R_2$", fontsize=12)
#                plt.xlabel("$t$", fontsize=12)
#                plt.ylim([-0.02, 1.02])
#
#                plt.tight_layout()
#
#                plt.show()
#
#            k += 1
#
#        m += 1
#
#    # r_matrix, R_matrix
#
#    return \
#        r1_matrix, phi1_matrix, r2_matrix, phi2_matrix, \
#        R1_matrix, Phi1_matrix, R2_matrix, Phi2_matrix, \
#        r_matrix, phi_matrix, R_matrix, Phi_matrix
# def get_synchro_transition_theta_2D(p_out_array, p_in, sizes, nb_instances,
#                                     t0, t1, dt, I_ext, I_array,
#                                     sigma, N, mean_SBM=False):
#     """
#     Predictions for the theta dynamics on modular graphs.
#
#     :param p_out_array: Array of probabilities, for a node, to be connected
#                         outside its community.
#   :param p_in: Probability, for a node, to be connected inside its community.
#                  If p_in = 0, it is a SBM graph.
#     :param sizes: [n1, n2] Community sizes
#     :param nb_instances: The number of instances of graphs
#                          and initial conditions
#     :param t0: Initial time
#     :param t1: Final time
#     :param dt: Time step
#     :param I_ext: 1xN : n1*[I1] + n2*[omega2]
#     :param I_array: 1x2 : [I1, I2]
#     :param sigma: Coupling constant
#     :param N: Number of nodes
#     :param mean_SBM: (Boolean)If we want to get the transition for the mean
#                      stochastic block model, it is True.
#
#     :return: Matrices nb_instances x len(p_out_array) measuring the phase
#              synchronization in each community and the global synchronization
#     """
#
#     n1, n2 = sizes
#
#     r1_matrix = np.zeros((nb_instances, len(p_out_array)))
#     phi1_matrix = np.zeros((nb_instances, len(p_out_array)))
#     r2_matrix = np.zeros((nb_instances, len(p_out_array)))
#     phi2_matrix = np.zeros((nb_instances, len(p_out_array)))
#     R1_matrix = np.zeros((nb_instances, len(p_out_array)))
#     Phi1_matrix = np.zeros((nb_instances, len(p_out_array)))
#     R2_matrix = np.zeros((nb_instances, len(p_out_array)))
#     Phi2_matrix = np.zeros((nb_instances, len(p_out_array)))
#     r_matrix = np.zeros((nb_instances, len(p_out_array)))
#     phi_matrix = np.zeros((nb_instances, len(p_out_array)))
#     R_matrix = np.zeros((nb_instances, len(p_out_array)))
#     Phi_matrix = np.zeros((nb_instances, len(p_out_array)))
#
#     m = 0
#
#     for p_out in tqdm(p_out_array):
#
#         k = 0
#
#         for instance in range(nb_instances):
#             time.sleep(3)
#
#             pq = [[p_in, p_out], [p_out, p_in]]
#
#             M_0 = np.block([[1/n1*np.ones(n1), np.zeros(n2)],
#                             [np.zeros(n1), 1/n2*np.ones(n2)]])
#             P = (np.block([[np.ones(n1), np.zeros(n2)],
#                            [np.zeros(n1), np.ones(n2)]])).T
#
#             if mean_SBM:
#                 A = np.zeros((N, N))
#                 ii = 0
#                 for i in range(0, len(sizes)):
#                     jj = 0
#                     for j in range(0, len(sizes)):
#                         A[ii:ii + sizes[i], jj:jj + sizes[j]] \
#                             = pq[i][j] * np.ones((sizes[i], sizes[j]))
#                         jj += sizes[j]
#                     ii += sizes[i]
#
#                 M = M_0
#
#             else:
#                 A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
#                 if p_in > 0:
#                     M = M_0
#                 else:
#                     V = get_eigenvectors_matrix(A, 2)  # Not normalized
#                     Vp = pinv(V)
#                     C = np.dot(M_0, Vp)
#                     CV = np.dot(C, V)
#                     M = (CV.T / (np.sum(CV, axis=1))).T
#
#             # Reduced paramters
#             redA = multi_dot([M, A, P])
#             hatredA = (multi_dot([M**2, A, P]).T / np.diag(np.dot(M, M.T))).T
#             hatLambda = multi_dot([M, A, pinv(M)])
#
#             # print("\n", redA, "\n", hatredA, "\n", hatLambda)
#
#             # import matplotlib.pyplot as plt
#             # plt.plot()
#             # plt.show()
#
#             theta0 = 2*np.pi*np.random.rand(N)
#             z0 = np.exp(1j * theta0)
#             Z0 = np.dot(M, z0)
#
#             # Integrate complete dynamics
#             args_theta = (I_ext, sigma)
#             theta_sol = integrate_dynamics(t0, t1, dt, theta, A,
#                                            "vode", theta0,
#                                            *args_theta)
#
#             z1 = np.sum(M[0, 0:n1] * np.exp(1j * theta_sol[:, 0:n1]), axis=1)
#             z2 = np.sum(M[1, n1:] * np.exp(1j * theta_sol[:, n1:]), axis=1)
#             z = 0.5 * (z1 + z2)
#
#             z1_mean = np.sum(z1[5 * int(t1 // dt) // 10:]
#                              ) / len(z1[5 * int(t1 // dt) // 10:])
#             z2_mean = np.sum(z2[5 * int(t1 // dt) // 10:]
#                              ) / len(z2[5 * int(t1 // dt) // 10:])
#             z_mean = np.sum(z[5 * int(t1 // dt) // 10:]
#                             ) / len(z[5 * int(t1 // dt) // 10:])
#
#             r1_mean, phi1_mean = np.absolute(z1_mean), np.angle(z1_mean)
#             r2_mean, phi2_mean = np.absolute(z2_mean), np.angle(z2_mean)
#             r_mean, phi_mean = np.absolute(z_mean), np.angle(z_mean)
#
#             # Integrate recuced dynamics
#             args_red_theta_2D = (I_array, sigma, N, hatredA, hatLambda)
#
#             red_theta_sol = integrate_dynamics(t0, t1, dt,
#                                                reduced_theta_2D,
#                                                redA, "zvode", Z0,
#                                                *args_red_theta_2D)
#
#             Z1, Z2 = red_theta_sol[:, 0], red_theta_sol[:, 1]
#             Z = 0.5 * (Z1 + Z2)
#
#             Z1_mean = np.sum(Z1[5 * int(t1 // dt) // 10:]
#                              ) / len(Z1[5 * int(t1 // dt) // 10:])
#             Z2_mean = np.sum(Z2[5 * int(t1 // dt) // 10:]
#                              ) / len(Z2[5 * int(t1 // dt) // 10:])
#             Z_mean = np.sum(Z[5 * int(t1 // dt) // 10:]
#                             ) / len(Z[5 * int(t1 // dt) // 10:])
#
#             R1_mean, Phi1_mean = np.absolute(Z1_mean), np.angle(Z1_mean)
#             R2_mean, Phi2_mean = np.absolute(Z2_mean), np.angle(Z2_mean)
#             R_mean, Phi_mean = np.absolute(Z_mean), np.angle(Z_mean)
#
#             r1_matrix[k, m], phi1_matrix[k, m] = r1_mean, phi1_mean
#             r2_matrix[k, m], phi2_matrix[k, m] = r2_mean, phi2_mean
#             R1_matrix[k, m], Phi1_matrix[k, m] = R1_mean, Phi1_mean
#             R2_matrix[k, m], Phi2_matrix[k, m] = R2_mean, Phi2_mean
#             r_matrix[k, m], phi_matrix[k, m] = r_mean, phi_mean
#             R_matrix[k, m], Phi_matrix[k, m] = R_mean, Phi_mean
#
#             # See the temporal series
#
#             # import matplotlib.pyplot as plt
#             #
#             # plt.subplot(311)
#             # plt.plot(r, color="k")
#             # plt.plot(R, color="g")
#             # plt.ylabel("$R$", fontsize=12)
#             #
#             # plt.subplot(312)
#             # plt.plot(r1, color="b")
#             # plt.plot(R1, color="lightblue")
#             # plt.ylabel("$R_1$", fontsize=12)
#             #
#             # plt.subplot(313)
#             # plt.plot(r2, color="r")
#             # plt.plot(R2, color="y")
#             # plt.ylabel("$R_2$", fontsize=12)
#             # plt.xlabel("$t$", fontsize=12)
#             # plt.tight_layout()
#             #
#             # plt.show()
#
#             k += 1
#
#         m += 1
#
#     return \
#         r1_matrix, phi1_matrix, r2_matrix, phi2_matrix, \
#         R1_matrix, Phi1_matrix, R2_matrix, Phi2_matrix, \
#         r_matrix, phi_matrix, R_matrix, Phi_matrix
