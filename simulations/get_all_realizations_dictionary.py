import networkx as nx
import time
from synch_predictions.graphs.get_reduction_matrix_and_characteristics import \
    get_realizations_dictionary, get_omega_realizations
import json
import numpy as np

nb_realizations = 50
dynamics_str = "kuramoto"  # "winfree", "kuramoto" or "theta"
parameter_realizations = True
n = 2

""" Adjacency matrix realizations parameters """
N1 = 150
N2 = 100
sizes = [N1, N2]
N = N1 + N2
p11 = 0.7
p22 = 0.5
p12 = 0.2
affinity_matrix = [[p11, p12],
                   [p12, p22]]
args_SBM = (sizes, affinity_matrix)
if p11 == 0 and p22 == 0:
    graph_str = "bipartite"
else:
    graph_str = "SBM"

print(f"Get realizations dictionary for {dynamics_str} on {graph_str} {n}D"
      f" with parameter_realizations = {parameter_realizations}.")


""" Omega realizations parameters """
if parameter_realizations:
    if dynamics_str == "theta":
        mean, std = 1.1, 0.01
    elif dynamics_str == "winfree":
        mean, std = 0.3, 0.01
    elif dynamics_str == "kuramoto":
        mean, std = 0.3, 0.01
    else:
        raise ValueError("Invalid dynamics_str, choose between winfree,"
                         " kuramoto and theta.")
    omega_realizations = \
        get_omega_realizations(nb_realizations, mean, std, N1, N2,
                               dynamics_str=dynamics_str)
    realizations_dictionary = \
        get_realizations_dictionary(omega_realizations, N1, N2,
                                    nb_realizations, nx.stochastic_block_model,
                                    args_SBM)
    realizations_dictionary["omega_realizations"] = omega_realizations
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    infos_realizations_dictionary = {"N1": N1,
                                     "N2": N2,
                                     "sizes": sizes,
                                     "affinity_matrix": affinity_matrix,
                                     "graph_str": graph_str,
                                     "number_realizations": nb_realizations,
                                     "omega_mean": mean,
                                     "omega_variance": variance,
                                     "timestr": timestr}
else:
    if dynamics_str == "theta":
        omega1 = -1.1
        omega2 = -0.9
    elif dynamics_str == "winfree":
        omega1 = 0.3
        omega2 = -N1/N2*omega1
    elif dynamics_str == "kuramoto":
        omega1 = 0.3
        omega2 = -N1/N2*omega1
    else:
        raise ValueError("Invalid dynamics_str, choose between winfree,"
                         " kuramoto and theta.")
    omega_realizations = []
    for i in range(nb_realizations):
        omega = np.concatenate([omega1*np.ones(N1), omega2*np.ones(N2)])
        omega_realizations.append(omega.tolist())
    realizations_dictionary = \
        get_realizations_dictionary(omega_realizations, N1, N2,
                                    nb_realizations, nx.stochastic_block_model,
                                    args_SBM)
    realizations_dictionary["omega_realizations"] = omega_realizations
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    infos_realizations_dictionary = {"N1": N1,
                                     "N2": N2,
                                     "sizes": sizes,
                                     "affinity_matrix": affinity_matrix,
                                     "graph_str": graph_str,
                                     "number_realizations": nb_realizations,
                                     "omega1": omega1,
                                     "omega2": omega2,
                                     "timestr": timestr}


with open(f'data/synchro_transitions_multiple_realizations/'
          f'{timestr}_realizations_dictionary_for_'
          f'{dynamics_str}_on_{graph_str}_{n}D_parameter_realizations'
          f'_{parameter_realizations}'                        
          f'.json', 'w') as outfile:
    json.dump(realizations_dictionary, outfile)


with open(f'data/synchro_transitions_multiple_realizations/'
          f'{timestr}_infos_realizations_dictionary_for_'
          f'{dynamics_str}_on_{graph_str}_{n}D_parameter_realizations'
          f'_{parameter_realizations}.json', 'w') as outfile:
    json.dump(infos_realizations_dictionary, outfile)
