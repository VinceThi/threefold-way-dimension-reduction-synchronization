from dynamics.get_synchro_transition import *
from graphs.get_reduction_matrix_and_characteristics import *
import json
import time

save_data = False
plot_time_series = True
dynamics_str = "kuramoto"  # "kuramoto" or "winfree" or "theta"
graph_str = "bipartite"    # "bipartite" or "SBM"
n = 2
number_sigma = 5
print(f"Simulation for the {dynamics_str} dynamics on the {graph_str} for "
      f"n = {n}. ")

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
    sigma_array = np.linspace(0, 15, number_sigma)
    # sigma_array = np.linspace(1, 5, number_sigma)
elif dynamics_str == "kuramoto" and graph_str == "SBM":
    sigma_array = np.linspace(0, 10, number_sigma)
elif dynamics_str == "theta" and graph_str == "bipartite":
    sigma_array = np.linspace(0, 10, number_sigma)
elif dynamics_str == "theta" and graph_str == "SBM":
    sigma_array = np.linspace(0, 10, number_sigma)
else:
    raise ValueError("Choose an appropriate sigma_array for your simulation.")

""" Integrate dynamics on network """
if dynamics_str == "kuramoto":
    synchro_transition_realizations_dictionary = \
        get_synchro_transition_realizations_phase_dynamics_random_graph(
            realizations_dictionary, kuramoto, reduced_kuramoto_complex,
            t0, t1, dt, averaging, sigma_array,
            plot_time_series=plot_time_series)
elif dynamics_str == "winfree":
    synchro_transition_realizations_dictionary = \
        get_synchro_transition_realizations_phase_dynamics_random_graph(
            realizations_dictionary, winfree, reduced_winfree_complex,
            t0, t1, dt, averaging, sigma_array,
            plot_time_series=plot_time_series)
elif dynamics_str == "theta":
    synchro_transition_realizations_dictionary = \
        get_synchro_transition_realizations_phase_dynamics_random_graph(
            realizations_dictionary, theta, reduced_theta_complex,
            t0, t1, dt, averaging, sigma_array,
            plot_time_series=plot_time_series)
else:
    raise ValueError("dynamics_str must be kuramoto, winfree, or theta.")

synchro_transition_realizations_dictionary["t0"] = t0
synchro_transition_realizations_dictionary["t1"] = t1
synchro_transition_realizations_dictionary["dt"] = dt
synchro_transition_realizations_dictionary["averaging"] = averaging

timestr = infos_realizations_dictionary["timestr"]

if save_data:

    with open(f'C:/Users/thivi/Documents/GitHub/network-synch/'
              f'synch_predictions/simulations/data/'
              f'synchro_transitions_multiple_realizations/'
              f'{timestr}_multiple_synchro_transition'
              f'_dictionary_{dynamics_str}_on_{graph_str}_{n}D'  # _naive'
              f'_{time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")}'
              f'.json', 'w') as outfile:
        json.dump(synchro_transition_realizations_dictionary, outfile)
