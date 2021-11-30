from synch_predictions.dynamics.get_synchro_transition import *
from synch_predictions.graphs.get_reduction_matrix_and_characteristics import *
from synch_predictions.plots.plot_complete_vs_reduced import *
import matplotlib.pyplot as plt
import json
import time

# Enter setup:
simulate = False
plot = True
save_data = True
plot_time_series = False
dynamics_str = "kuramoto"
graph_str = "bipartite"
n = 2
number_sigma = 50
print(f"Simulation for the {dynamics_str} dynamics on the {graph_str} for "
      f"n = {n}. ")

if simulate:
    """ Time parameters """
    t0, t1, dt = 0, 10000, 0.4
    time_list = np.linspace(t0, t1, int(t1 / dt))
    averaging = 5

    """ Structural parameters """
    realizations_dictionary_str = f"{graph_str}_{dynamics_str}"
    realizations_dictionary_path = \
        get_realizations_dictionary_absolute_path(realizations_dictionary_str)
    with open(f'C:/Users/thivi/Documents/GitHub/network-synch/'
              f'synch_predictions/simulations/data/'
              f'synchro_transitions_multiple_realizations/'
              f'{realizations_dictionary_path}.json') as json_data:
        realizations_dictionary = json.load(json_data)
    infos_realizations_dictionary_str = f"{graph_str}_{dynamics_str}"
    infos_realizations_dictionary_path = \
        get_infos_realizations_dictionary_absolute_path(
            infos_realizations_dictionary_str)
    with open(f'C:/Users/thivi/Documents/GitHub/network-synch/'
              f'synch_predictions/simulations/data/'
              f'synchro_transitions_multiple_realizations/'
              f'{infos_realizations_dictionary_path}.json') as json_data:
        infos_realizations_dictionary = json.load(json_data)

    """ Dynamical parameters """
    if dynamics_str == "winfree" and graph_str == "bipartite":
        sigma_array = np.linspace(0, 10, number_sigma)
    elif dynamics_str == "winfree" and graph_str == "SBM":
        sigma_array = np.linspace(0, 5, number_sigma)
    elif dynamics_str == "kuramoto" and graph_str == "bipartite":
        # sigma_array = np.linspace(0, 15, number_sigma)
        sigma_array = np.linspace(3, 9, number_sigma)
    elif dynamics_str == "kuramoto" and graph_str == "SBM":
        sigma_array = np.linspace(0, 10, number_sigma)
    elif dynamics_str == "theta" and graph_str == "bipartite":
        sigma_array = np.linspace(0, 10, number_sigma)
    elif dynamics_str == "theta" and graph_str == "SBM":
        sigma_array = np.linspace(0, 10, number_sigma)
    else:
        raise ValueError("Choose an appropriate sigma_array"
                         " for your simulation.")

    """ Integrate dynamics on network """
    adjacency_matrix_realizations = \
        realizations_dictionary["adjacency_matrix_realizations"]
    M_realizations = \
        realizations_dictionary["M_realizations"]
    omega_realizations = \
        realizations_dictionary["omega_realizations"]

    A = np.array(adjacency_matrix_realizations[0])
    M = np.array(M_realizations[0])

    # print(M@A@np.linalg.pinv(M))

    W = np.array(np.diag(omega_realizations[0]))
    theta0 = 2*np.pi*np.random.randn(len(A[:, 0]))
    synchro_transition_dictionary = \
        get_synchro_transition_phase_dynamics_graphs_backward_forward(
            kuramoto, reduced_kuramoto_complex, t0, t1, dt, averaging,
            "A", "W", "None", theta0, M, sigma_array, W, A,
            plot_time_series=False)

    synchro_transition_dictionary["t0"] = t0
    synchro_transition_dictionary["t1"] = t1
    synchro_transition_dictionary["dt"] = dt
    synchro_transition_dictionary["averaging"] = averaging

    # if dynamics_str == "kuramoto":
    # synchro_transition_realizations_dictionary = \
    #     get_synchro_transition_realizations_phase_dynamics_random_graph(
    #         realizations_dictionary, kuramoto, reduced_kuramoto_complex,
    #         t0, t1, dt, averaging, sigma_array,
    #         plot_time_series=plot_time_series)
    # elif dynamics_str == "winfree":
    #     synchro_transition_realizations_dictionary = \
    #         get_synchro_transition_realizations_phase_dynamics_random_graph(
    #             realizations_dictionary, winfree, reduced_winfree_complex,
    #             t0, t1, dt, averaging, sigma_array,
    #             plot_time_series=plot_time_series)
    # elif dynamics_str == "theta":
    #     synchro_transition_realizations_dictionary = \
    #         get_synchro_transition_realizations_phase_dynamics_random_graph(
    #             realizations_dictionary, theta, reduced_theta_complex,
    #             t0, t1, dt, averaging, sigma_array,
    #             plot_time_series=plot_time_series)
    # else:
    #     raise ValueError("dynamics_str must be kuramoto, winfree, or theta.")

    timestr = infos_realizations_dictionary["timestr"]
       
    if save_data:

        with open(f'C:/Users/thivi/Documents/GitHub/network-synch/'
                  f'synch_predictions/simulations/data/'
                  f'synchro_transitions_multiple_realizations/'
                  f'{timestr}_one_synchro_transition_backward_forward'
                  f'_dictionary_{dynamics_str}_on_{graph_str}_{n}D'
                  f'_{time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")}'
                  f'.json', 'w') as outfile:
            json.dump(synchro_transition_dictionary, outfile)

else:
    file = "2020_02_19_20h39min44sec_one_synchro_transition" \
           "_backward_forward_dictionary_kuramoto_on_bipartite" \
           "_2D_2020_02_21_17h57min12sec"
    with open("data/synchro_transitions_multiple_realizations/"
              + file + ".json") \
            as json_data:
        synchro_transition_dictionary = json.load(json_data)
    sigma_array = synchro_transition_dictionary["sigma_array"]
    r_array_f = synchro_transition_dictionary["r_forward"]
    R_array_f = synchro_transition_dictionary["R_forward"]
    r_mu_matrix_array_f = synchro_transition_dictionary["r_mu_matrix_forward"]
    R_mu_matrix_array_f = synchro_transition_dictionary["R_mu_matrix_forward"]

    r_array_b = synchro_transition_dictionary["r_backward"]
    R_array_b = synchro_transition_dictionary["R_backward"]
    r_mu_matrix_array_b = synchro_transition_dictionary["r_mu_matrix_backward"]
    R_mu_matrix_array_b = synchro_transition_dictionary["R_mu_matrix_backward"]

    plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    plot_transitions_complete_vs_reduced_one_instance(ax, sigma_array,
                                                      r_array_f,
                                                      R_array_f, "k",
                                                      "#969696",
                                                      0.9, "o", 100, 2)
    plot_transitions_complete_vs_reduced_one_instance(ax, sigma_array,
                                                      r_array_b,
                                                      R_array_b, "r", "y",
                                                      0.9, "o", 20, 2)
    plt.show()
