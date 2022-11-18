import networkx as nx
import time
from synch_predictions.graphs.get_reduction_matrix_and_characteristics import \
    get_random_graph_realizations
import json

number_realizations = 10000
N1 = 150
N2 = 100
sizes = [N1, N2]
N = N1 + N2
p11 = 0
p22 = 0
p12 = 0.2
p21 = 0.2
file = "SBM_p11_0_p22_0_pout_0_2"
affinity_matrix = [[p11, p12],
                   [p21, p22]]

args_SBM = (sizes, affinity_matrix)
if p11 == 0 and p22 == 0:
    graph_str = "bipartite"
else:
    graph_str = "SBM"

SBM_instance_infos_dictionary = {"N1": N1,
                                 "N2": N2,
                                 "sizes": sizes,
                                 "affinity_matrix": affinity_matrix,
                                 "graph_str": graph_str,
                                 "number_realizations": number_realizations}

A_list = \
    get_random_graph_realizations(nx.stochastic_block_model,
                                  number_realizations, file, *args_SBM)

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

with open(f'SBM_instances/{file}/'
          f'{timestr}_SBM_realizations_info_dictionary'
          f'_{graph_str}_p11_{p11}_p22_{p22}_p12_{p12}_N1_{N1}_N2_{N2}'
          f'.json', 'w') as outfile:
    json.dump(SBM_instance_infos_dictionary, outfile)
