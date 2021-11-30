from dynamicalab.generators import connectomes_CElegansSNAP
# from dynamicalab.generators import plants_pollinators_McCullen1993
from synch_predictions.plots.plot_spectrum import plot_spectrum,\
    plot_complex_spectrum
import networkx as nx
import collections
import dynamicalab.drawing as draw
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import numpy as np
import pandas as pd
import json
import csv


reduced_first_community_color = "#9ecae1"
reduced_second_community_color = "#fdd0a2"
reduced_third_community_color = "#a1d99b"
reduced_fourth_community_color = "#9e9ac8"


def print_network_properties(animal_string, graph, adjacency_matrix):
    N = len(adjacency_matrix[:, 0])
    M = graph.number_of_edges()
    in_degree_sequence = np.sum(adjacency_matrix, axis=1)
    out_degree_sequence = np.sum(adjacency_matrix, axis=0)
    mean_degree = np.mean(np.array(in_degree_sequence))
    # out_mean_degree = np.mean(np.array(out_degree_sequence))
    max_in_degree = max(in_degree_sequence)
    max_out_degree = max(out_degree_sequence)
    print(f"\n\n{animal_string}\n",
          f"Number of nodes N = {N}\n",
          f"Number of edges M = {M}\n",
          f"<k_in> = <k_out>= {mean_degree}\n",
          f"k_in_max = {int(max_in_degree)}\n"                      
          f"k_out_max = {int(max_out_degree)}\n")


# Zebrafish larvaire
# """
# Kunst et al. "A Cellular-Resolution Atlas of the Larval Zebrafish Brain",
# (2019) avec le traitement de Antoine Légaré
# On a pas exactement les mêmes régions que l'article non plus,
#  où la matrice est faite avec 36 régions. Ici, on en a 71 qui sont
#  mutually exclusive et collectively exhaustive (je reprends les
# termes du dude dans le courriel) donc ça couvre tout le volume au
#  complet sans overlap

path_str = "C:/Users/thivi/Documents/GitHub/network-synch/" \
           "synch_predictions/plots/data/"

df = pd.read_csv(path_str +
                 'Connectivity_matrix_zebra_fish_mesoscopic.csv')
dictio = {'X': 0}  # We put zeros temporarily on the diagonal
df = df.replace(dictio)

volumes = np.array(1 * np.load(path_str + "volumes_zebrafish_meso.npy"))
relativeVolumes = volumes / sum(volumes)
adjacency = df.to_numpy()[:, 1:-1].astype(float)
N = len(adjacency[0])
# """ To get an undirected graph """
# for i in range(adjacency.shape[0]):
#     for j in range(i+1, adjacency.shape[0]):
#         adjacency[i, j] = (adjacency[i, j] + adjacency[j, i]) /
#  (relativeVolumes[i] + relativeVolumes[j])
#         adjacency[j, i] = adjacency[i, j]
""" To get a directed graph """
for i in range(adjacency.shape[0]):
    for j in range(adjacency.shape[0]):
        adjacency[i, j] = adjacency[i, j] / (
                relativeVolumes[i] + relativeVolumes[j])
adjacency = adjacency / np.amax(adjacency)
adjacency = np.log(adjacency + 0.00001)
adjacency -= np.amin(adjacency)
adjacency = adjacency / np.amax(adjacency)
A_zebra = adjacency + np.eye(N)

# N = 71
# rank_zebrafish_meso = 71
G_zebra = nx.from_numpy_array(A_zebra)
# print(np.linalg.matrix_rank(A_zebra))
print(G_zebra.number_of_nodes())
print(G_zebra.number_of_edges())
# print(G_zebra.is_directed())    # They don't return the good thing ...
# print(nx.is_weighted(G_ciona))  # They don't return the good thing ...
# degree_values = list(dict(G_ciona.degree).values())
# degree_array = np.array(degree_values)
# print(np.mean(degree_array))


U, S, Vh = np.linalg.svd(A_zebra)
# Ur, Sr, Vhr = svds(D_no_d, 1)
rank_D = np.linalg.matrix_rank(A_zebra)
n = 8
print(f"rank_D = {rank_D}")
print(f"n = {n} \n")
Ur = U[:, :n]
Sr = np.diag(S[:n])
Vhr = Vh[:n, :]
DVhr = np.diag(-(np.sum(Vhr, axis=1) < 0).astype(float)) \
       + np.diag((np.sum(Vhr, axis=1) >= 0).astype(float))
M = np.sqrt(Sr)@DVhr@Vhr

A_zebra_reduced = M@A_zebra@np.linalg.pinv(M)
G_zebra_reduced = nx.from_numpy_array(A_zebra_reduced)


A = A_zebra_reduced    # _reduced
G = G_zebra_reduced    # _reduced
k = np.sum(A, axis=1)
pos = nx.kamada_kawai_layout(G_zebra)
# pos = nx.circular_layout(G)
# pos = nx.shell_layout(G)
# pos = nx.layout.spectral_layout(G)
# pos = nx.spring_layout(G)
pos = nx.spring_layout(G, pos=pos, iterations=1000)

node_colors = reduced_first_community_color

sns.set(style="ticks")

plt.figure(figsize=(3, 3))
ax = plt.gca()
draw.draw_networks(G, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=2,
                   edge_alpha=0.4,
                   use_edge_weigth=True,
                   node_width=1,
                   node_size=25.0,  # 30*k
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()
# """



# Ciona intestinalis lavaire
"""
# https://doi.org/10.7554/eLife.16962.040
# The CNS connectome of a tadpole larva of Ciona intestinalis (L.) 
# highlights sidedness in the brain of a chordate sibling
# - Kerrianne Ryan, Zhiyuan Lu, Ian A Meinertzhagen
# 
# For the dataset with all pre and post synaptic cells: 
# ciona_intestinalis_lavaire_elife-16962-fig16-data1-v1
# 
# Also included are muscle cells,and the basal lamina of the CNS, both of which
# are exclusively postsynaptic. Other cell types, particularly ependymal cells
# lacking axons, are excluded. Muscle cells of the dorsal and medial bands are 
# pooled on each side, because these are connected via gap junctions
# => the dataset contains muscle(outputs,6 last columns of the dataset I think)
# 
# The rows are presynaptic cells and the columns are postsynaptic cells.
# 
# See Table 3. Numbers of cells in the 
# left, right and centre of the CNS and PNS.
# 
# 178 CNS and 28 PNS = 206 neurons 
A_from_xlsx = pd.read_excel('data/ciona_intestinalis_lavaire_elife'
                            '-16962-fig16-data1-v1_modified.xlsx').values
A_ciona_nan = np.array(A_from_xlsx[0:, 1:])
A_ciona = np.array(A_ciona_nan, dtype=float)
where_are_NaNs = np.isnan(A_ciona)
A_ciona[where_are_NaNs] = 0
print(np.linalg.eig(A_ciona))
print(A_ciona)
print(np.shape(A_ciona))
G_ciona = nx.from_numpy_array(A_ciona)
# print(np.linalg.matrix_rank(A_ciona))
print(G_ciona.number_of_nodes())
print(G_ciona.number_of_edges())
# print(G_ciona.is_directed())    # They don't return the good thing ...
# print(nx.is_weighted(G_ciona))  # They don't return the good thing ...
# degree_values = list(dict(G_ciona.degree).values())
# degree_array = np.array(degree_values)
# print(np.mean(degree_array))

pos = nx.kamada_kawai_layout(G_ciona)
# pos = nx.circular_layout(G_ciona)
# pos = nx.shell_layout(G_ciona)
# pos = nx.layout.spectral_layout(G_ciona)
# pos = nx.spring_layout(G_ciona)
pos = nx.spring_layout(G_ciona, pos=pos, iterations=1000)

node_colors = reduced_third_community_color

sns.set(style="ticks")
plt.figure(figsize=(3, 3))
ax = plt.gca()
draw.draw_networks(G_ciona, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.4,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=15.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()
"""


# C. elegans
"""
# Data obtained from Mohamed Bahdine, extracted as described in the
# supplementary material of the article : Network control principles 
# predict neuron function in the C. elegans connectome - Yan,..., Barabasi
# The data come from Wormatlas.

A_celegans = np.array(1*np.load("data/C_Elegans.npy"))
G_celegans = nx.from_numpy_array(A_celegans)
# print(A_celegans[0:20, 0:20])
# print(np.linalg.eig(A_celegans))  # They are complex !
print(np.linalg.matrix_rank(A_celegans))
print(G_celegans.number_of_nodes())
print(G_celegans.number_of_edges())
print(G_celegans.is_directed())    # They return the contrary ... ?
print(nx.is_weighted(G_celegans))  # They return the contrary ... ?
degree_values = list(dict(G_celegans.degree).values())
degree_array = np.array(degree_values)
print(np.mean(degree_array))
pos = nx.kamada_kawai_layout(G_celegans)

pos = nx.kamada_kawai_layout(G_celegans)
# pos = nx.circular_layout(G_mouse)
# pos = nx.shell_layout(G_mouse)
# pos = nx.layout.spectral_layout(G_mouse)
# pos = nx.spring_layout(G_mouse)
pos = nx.spring_layout(G_celegans, pos=pos, iterations=1000)

node_colors = reduced_third_community_color

sns.set(style="ticks")
plt.figure(figsize=(3, 3))
ax = plt.gca()
draw.draw_networks(G_celegans, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.4,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=15.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()
"""


# Mouse visual cortex (see Fig. 5 paper Bock 2011 Nature)
"""
# @inproceedings{nr,
#      title={The Network Data Repository
#             with Interactive Graph Analytics and Visualization},
#      author={Ryan A. Rossi and Nesreen K. Ahmed},
#      booktitle={AAAI},
#      url={http://networkrepository.com},
#      year={2015}
# }

G_mouse = nx.Graph()
edges = nx.read_edgelist('data/bn-mouse_visual-cortex_2.edges')
G_mouse.add_edges_from(edges.edges())
A_mouse = nx.to_numpy_array(G_mouse)
print(np.linalg.eig(A_mouse))  # Elles sont réelles !
print(np.linalg.matrix_rank(A_mouse))
print(G_mouse.number_of_nodes())
print(G_mouse.number_of_edges())
print(G_mouse.is_directed())
print(nx.is_weighted(G_mouse))
degree_values = list(dict(G_mouse.degree).values())
degree_array = np.array(degree_values)
print(np.mean(degree_array))
pos = nx.kamada_kawai_layout(G_mouse)
# pos = nx.circular_layout(G_mouse)
# pos = nx.shell_layout(G_mouse)
# pos = nx.layout.spectral_layout(G_mouse)
# pos = nx.spring_layout(G_mouse)
pos = nx.spring_layout(G_mouse, pos=pos, iterations=1000)

# node_colors = np.random.choice([reduced_first_community_color,
#                                 reduced_second_community_color,
#                                 reduced_third_community_color,
#                                 reduced_fourth_community_color],
#                                30, replace=True)
node_colors = reduced_third_community_color

sns.set(style="ticks")
plt.figure(figsize=(5, 5))
ax = plt.gca()
draw.draw_networks(G_mouse, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.4,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=15.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()
"""


# Mouse mesoscopic connectome
"""
# Oh, S., Harris, J., Ng, L. et al. A mesoscale connectome of the mouse brain.
#  Nature 508, 207–214 (2014) doi:10.1038/nature13186

A_mouse_meso = np.loadtxt("data/ABA_weight_mouse.txt")
# print(np.linalg.eig(A_mouse_meso))  # They are complex !
# print(A_mouse_meso)
# print(np.shape(A_mouse_meso))
G_mouse_meso = nx.from_numpy_array(A_mouse_meso)
# print(np.linalg.matrix_rank(A_mouse_meso))
print(G_mouse_meso.number_of_nodes())
print(G_mouse_meso.number_of_edges())
# print(G_mouse_meso.is_directed())    # They don't return the good thing ...
# print(nx.is_weighted(G_mouse_meso))  # They don't return the good thing ...
degree_values = list(dict(G_mouse_meso.degree).values())
degree_array = np.array(degree_values)
print(np.mean(degree_array))


path = "C:/Users/thivi/Documents/GitHub/" \
       "network-synch/synch_predictions/plots/data/"
A_mouse_meso_binary = np.array(list(csv.reader(
    open(path+"mouse_connectome-Oh_Nature_2014.csv", "rt"), delimiter=",")),
    dtype=float)
# print(np.linalg.eig(A_mouse_meso_binary))  # They are complex !
# print(A_mouse_meso_binary)
# print(np.shape(A_mouse_meso))
G_mouse_meso_binary = nx.from_numpy_array(A_mouse_meso_binary)
# print(np.linalg.matrix_rank(A_mouse_meso))
print(G_mouse_meso_binary.number_of_nodes())
print(G_mouse_meso_binary.number_of_edges())
# print(G_mouse_meso.is_directed())    # They don't return the good thing ...
# print(nx.is_weighted(G_mouse_meso))  # They don't return the good thing ...
degree_values = list(dict(G_mouse_meso_binary.degree).values())
degree_array = np.array(degree_values)
print(np.mean(degree_array))

print(type(A_mouse_meso_binary))

fig = plt.figure(figsize=(6, 4))
ax1 = plt.subplot(121)
plt.gca()
ax1.matshow(A_mouse_meso>0, aspect="auto")
ax2 = plt.subplot(122)
plt.gca()
ax2.matshow(A_mouse_meso_binary, aspect="auto")
plt.show()

pos = nx.kamada_kawai_layout(G_mouse_meso)
# pos = nx.circular_layout(G_mouse_meso)
# pos = nx.shell_layout(G_mouse_meso)
# pos = nx.layout.spectral_layout(G_mouse_meso)
# pos = nx.spring_layout(G_mouse_meso)
pos = nx.spring_layout(G_mouse_meso, pos=pos, iterations=1000)

node_colors = reduced_third_community_color

sns.set(style="ticks")
plt.figure(figsize=(3, 3))
ax = plt.gca()
draw.draw_networks(G_mouse_meso_binary, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.4,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=15.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()


"""


# Drosophila melanogaster
"""
df = pd.read_csv('data/drosophila_exported-traced-adjacencies-v1.1/'
                 'traced-total-connections.csv')
Graphtype = nx.DiGraph()
G_drosophila = nx.from_pandas_edgelist(df,
                                       source='bodyId_pre',
                                       target='bodyId_post',
                                       edge_attr='weight',
                                       create_using=Graphtype)
# A_drosophila = nx.to_numpy_array(G_drosophila)

pos = nx.kamada_kawai_layout(G_drosophila)
# pos = nx.circular_layout(G_drosophila)
# pos = nx.shell_layout(G_drosophila)
# pos = nx.layout.spectral_layout(G_drosophila)
# pos = nx.spring_layout(G_drosophila)
pos = nx.spring_layout(G_drosophila, pos=pos, iterations=1000)

node_colors = reduced_third_community_color

sns.set(style="ticks")
plt.figure(figsize=(3, 3))
ax = plt.gca()
draw.draw_networks(G_drosophila, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.4,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=15.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()
"""


# Macaque
"""
# @inproceedings{nr,
#      title={The Network Data Repository
#             with Interactive Graph Analytics and Visualization},
#      author={Ryan A. Rossi and Nesreen K. Ahmed},
#      booktitle={AAAI},
#      url={http://networkrepository.com},
#      year={2015}
# }

G_macaque = nx.Graph()
edges = nx.read_edgelist('data/bn-macaque-rhesus_brain_1.edges')
G_macaque.add_edges_from(edges.edges())
# pos = nx.kamada_kawai_layout(G_macaque)
# pos = nx.circular_layout(G_macaque)
# pos = nx.shell_layout(G_macaque)
pos = nx.layout.spectral_layout(G_macaque)
# pos = nx.spring_layout(G_macaque)
pos = nx.spring_layout(G_macaque, pos=pos, iterations=1000)
reduced_first_community_color = "#9ecae1"
reduced_second_community_color = "#fdd0a2"
reduced_third_community_color = "#a1d99b"
reduced_fourth_community_color = "#9e9ac8"

# node_colors = np.random.choice([reduced_first_community_color,
#                                 reduced_second_community_color,
#                                 reduced_third_community_color,
#                                 reduced_fourth_community_color],
#                                30, replace=True)
node_colors = reduced_third_community_color

sns.set(style="ticks")
plt.figure(figsize=(3, 3))
ax = plt.gca()
draw.draw_networks(G_macaque, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.2,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=20.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()
"""


# Power bus
"""
# @inproceedings{nr,
#      title={The Network Data Repository
#             with Interactive Graph Analytics and Visualization},
#      author={Ryan A. Rossi and Nesreen K. Ahmed},
#      booktitle={AAAI},
#      url={http://networkrepository.com},
#      year={2015}
# }

adjacency_matrix = sio.mmread('data/power-494-bus.mtx').toarray()

G_powerbus = nx.from_numpy_matrix(adjacency_matrix)

# G_power.add_edges_from(edges.edges())
# pos = nx.kamada_kawai_layout(G_powerbus)
# pos = nx.circular_layout(G_powerbus)
# pos = nx.shell_layout(G_powerbus)
pos = nx.layout.spectral_layout(G_powerbus)
pos = nx.spring_layout(G_powerbus)
# pos = nx.spring_layout(G_powerbus, pos=pos, iterations=1000)
reduced_first_community_color = "#9ecae1"
reduced_second_community_color = "#fdd0a2"
reduced_third_community_color = "#a1d99b"
reduced_fourth_community_color = "#9e9ac8"

# node_colors = np.random.choice([reduced_first_community_color,
#                                 reduced_second_community_color,
#                                 reduced_third_community_color,
#                                 reduced_fourth_community_color],
#                                30, replace=True)
node_colors = reduced_first_community_color

sns.set(style="ticks")
plt.figure(figsize=(6, 6))
ax = plt.gca()
draw.draw_networks(G_powerbus, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.2,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=20.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()
"""


# Power US grid
"""
# Huge data, too long to visualize
# @inproceedings{nr,
#      title={The Network Data Repository
#             with Interactive Graph Analytics and Visualization},
#      author={Ryan A. Rossi and Nesreen K. Ahmed},
#      booktitle={AAAI},
#      url={http://networkrepository.com},
#      year={2015}
# }

adjacency_matrix = sio.mmread('data/power-US-Grid.mtx').toarray()

G_power = nx.from_numpy_matrix(adjacency_matrix)
# edges = nx.read_edgelist('data/power-US-Grid.mtx')
# G_power.add_edges_from(edges.edges())
# pos = nx.kamada_kawai_layout(G_power)
# pos = nx.circular_layout(G_power)
# pos = nx.shell_layout(G_power)
pos = nx.layout.spectral_layout(G_power)
# pos = nx.spring_layout(G_power)
pos = nx.spring_layout(G_power, pos=pos, iterations=1000)
reduced_first_community_color = "#9ecae1"
reduced_second_community_color = "#fdd0a2"
reduced_third_community_color = "#a1d99b"
reduced_fourth_community_color = "#9e9ac8"

# node_colors = np.random.choice([reduced_first_community_color,
#                                 reduced_second_community_color,
#                                 reduced_third_community_color,
#                                 reduced_fourth_community_color],
#                                30, replace=True)
node_colors = reduced_first_community_color

sns.set(style="ticks")
plt.figure(figsize=(3, 3))
ax = plt.gca()
draw.draw_networks(G_power, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.2,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=20.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()
"""


# C. elegans frontal
"""
G_celegans = connectomes_CElegansSNAP().graph()
pos = nx.kamada_kawai_layout(G_celegans)
# pos = nx.circular_layout(G_celegans)
# pos = nx.shell_layout(G_celegans)
# pos = nx.layout.spectral_layout(G_celegans)
# pos = nx.spring_layout(G_celegans, pos=pos, iterations=1000)
reduced_first_community_color = "#9ecae1"
reduced_second_community_color = "#fdd0a2"
reduced_third_community_color = "#a1d99b"
reduced_fourth_community_color = "#9e9ac8"

# node_colors = np.random.choice([reduced_first_community_color,
#                                 reduced_second_community_color,
#                                 reduced_third_community_color,
#                                 reduced_fourth_community_color],
#                                30, replace=True)
node_colors = reduced_third_community_color

sns.set(style="ticks")
plt.figure(figsize=(8, 8))
ax = plt.gca()
draw.draw_networks(G_celegans, pos, ax,
                   mu=0.08,
                   edge_color="#b3b3b3",
                   edge_width=1.5,
                   edge_alpha=0.6,
                   use_edge_weigth=False,
                   node_width=1.5,
                   node_size=100.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()

"""

# Food web
"""
G_foodweb = plants_pollinators_McCullen1993().graph
pos = nx.spring_layout(G_celegans)

sns.set(style="ticks")
fig = plt.figure()
ax = plt.gca()

node_colors = np.random.choice(['#b2182b',
                                '#d6604d',
                                '#f4a582',
                                '#fddbc7',
                                '#f7f7f7',
                                '#d1e5f0',
                                '#92c5de',
                                '#4393c3',
                                '#2166ac'],  20, replace=True)

draw.draw_networks(G_foodweb, pos, ax,
                   mu=0.08,
                   edge_color="#b3b3b3",
                   edge_width=1.5,
                   edge_alpha=0.6,
                   use_edge_weigth=False,
                   node_width=1.5,
                   node_size=100.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.show()

"""


# Degree histogram
"""

# adjacency_matrix = sio.mmread('data/power-494-bus.mtx').toarray()
# G = nx.from_numpy_matrix(adjacency_matrix)
A_tree = np.array([[0, 1, 0, 0, 0, 0, 0],
                   [1, 0, 1, 1, 0, 0, 0],
                   [0, 1, 0, 0, 1, 1, 0],
                   [0, 1, 0, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0]])
G_tree = nx.from_numpy_matrix(A_tree)
degree_sequence_tree = sorted([d for n, d in G_tree.degree()], reverse=True)
degreeCount_tree = collections.Counter(degree_sequence_tree)
deg_tree, cnt_tree = zip(*degreeCount_tree.items())

G_mouse = nx.Graph()
edges = nx.read_edgelist('data/bn-mouse_visual-cortex_2.edges')
G_mouse.add_edges_from(edges.edges())
degree_sequence_mouse = sorted([d for n, d in G_mouse.degree()], reverse=True)
degreeCount_mouse = collections.Counter(degree_sequence_mouse)
deg_mouse, cnt_mouse = zip(*degreeCount_mouse.items())

A_celegans = np.load("data/C_Elegans.npy")
G = nx.from_numpy_array(A_celegans)
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())


plt.figure(figsize=(6.5, 4))

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True

plt.subplot(231)
pos = nx.kamada_kawai_layout(G_tree)
# pos = nx.circular_layout(G_tree)
# pos = nx.shell_layout(G_tree)
# pos = nx.layout.spectral_layout(G_tree)
# pos = nx.spring_layout(G_tree)
pos = nx.spring_layout(G_tree, pos=pos, iterations=1000)
# node_colors = np.random.choice([reduced_first_community_color,
#                                 reduced_second_community_color,
#                                 reduced_third_community_color,
#                                 reduced_fourth_community_color],
#                                30, replace=True)
node_colors = reduced_first_community_color
sns.set(style="ticks")
ax = plt.gca()
draw.draw_networks(G_tree, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.4,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=15.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.subplot(232)
pos = nx.kamada_kawai_layout(G)
# pos = nx.circular_layout(G_mouse)
# pos = nx.shell_layout(G_mouse)
# pos = nx.layout.spectral_layout(G_mouse)
# pos = nx.spring_layout(G_mouse)
pos = nx.spring_layout(G, pos=pos, iterations=1000)

node_colors = reduced_second_community_color

sns.set(style="ticks")
ax = plt.gca()
draw.draw_networks(G, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.4,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=15.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.subplot(233)
pos = nx.kamada_kawai_layout(G_mouse)
# pos = nx.circular_layout(G_mouse)
# pos = nx.shell_layout(G_mouse)
# pos = nx.layout.spectral_layout(G_mouse)
# pos = nx.spring_layout(G_mouse)
pos = nx.spring_layout(G_mouse, pos=pos, iterations=1000)
# node_colors = np.random.choice([reduced_first_community_color,
#                                 reduced_second_community_color,
#                                 reduced_third_community_color,
#                                 reduced_fourth_community_color],
#                                30, replace=True)
node_colors = reduced_third_community_color
sns.set(style="ticks")
ax = plt.gca()
draw.draw_networks(G_mouse, pos, ax,
                   mu=0.02,
                   edge_color="#b3b3b3",
                   edge_width=1,
                   edge_alpha=0.4,
                   use_edge_weigth=False,
                   node_width=1,
                   node_size=15.0,
                   node_border_color="#3f3f3f",
                   node_color=node_colors,
                   node_alpha=1.0,
                   arrow_scale=0.0,
                   loop_radius=0)

plt.subplot(234)
plt.bar(deg_tree, cnt_tree, width=0.20, color=reduced_first_community_color)
# plt.title("Degree Histogram")
plt.ylabel("Compte")
plt.xlabel("Degré $k$")
# ax.set_xticks([d + 0.4 for d in deg])
# ax.set_xticklabels(deg)

plt.subplot(235)
plt.bar(deg, cnt, width=3, color=reduced_second_community_color)
# plt.title("Degree Histogram")
# plt.ylabel("Compte")
plt.xlabel("Degré $k$")
# ax.set_xticks([d + 0.4 for d in deg])
# ax.set_xticklabels(deg)
plt.xlim([0, 100])
# plt.ylim([0, 30])
plt.yticks([0, 10, 20, 30])
plt.yscale("log")
# plt.xscale("log")

plt.subplot(236)
plt.bar(deg_mouse, cnt_mouse, width=1, color=reduced_third_community_color)
# plt.title("Degree Histogram")
# plt.ylabel("Compte")
plt.xlabel("Degré $k$")
# ax.set_xticks([d + 0.4 for d in deg])
# ax.set_xticklabels(deg)
plt.xlim([0, 30])
# plt.ylim([0, 30])
plt.xticks([1, 10, 20, 30])
plt.yscale("log")
# plt.xscale("log")


plt.tight_layout()

plt.show()

"""


# Spectrum
"""
# A = sio.mmread('data/power-494-bus.mtx').toarray()

# G = nx.Graph()
# edges = nx.read_edgelist('data/bn-mouse_visual-cortex_2.edges')
# G.add_edges_from(edges.edges())
# A = nx.to_numpy_matrix(G)

# A_star = nx.to_numpy_array(nx.star_graph(5))
# # print(np.linalg.eig(A_star))
A_celegans = np.load("data/C_Elegans.npy")
G_celegans = nx.from_numpy_array(A_celegans)
print_network_properties("Caenorhabditis elegans", G_celegans, A_celegans)
# Caenorhabditis elegans
# Number of nodes N = 279
# Number of edges M = 1999
# <k_in> = <k_out>= 11.157706093189963
# k_in_max = 61
# k_out_max = 85


A_from_xlsx = pd.read_excel('data/ciona_intestinalis_lavaire_elife'        
                            '-16962-fig16-data1-v1_modified.xlsx').values
A_ciona_nan = np.array(A_from_xlsx[0:, 1:])
A_ciona = np.array(A_ciona_nan, dtype=float)
where_are_NaNs = np.isnan(A_ciona)
A_ciona[where_are_NaNs] = 0
# A_ciona = A_ciona > 0
G_ciona = nx.from_numpy_array(A_celegans)
print_network_properties("Ciona intestinalis", G_ciona, A_ciona)


df = pd.read_csv('data/drosophila_exported-traced-adjacencies-v1.1/'
                 'traced-total-connections.csv')
Graphtype = nx.DiGraph()
G_drosophila = nx.from_pandas_edgelist(df,
                                       source='bodyId_pre',
                                       target='bodyId_post',
                                       edge_attr='weight',
                                       create_using=Graphtype)
A_drosophila = nx.to_numpy_array(G_drosophila)
A_drosophila_binary = (A_drosophila >  # <k_in> = 162,  k_in_max = 5124
                       np.zeros(np.shape(A_drosophila))).astype(float)
G_drosophila_binary = nx.from_numpy_array(A_drosophila_binary)
# print_network_properties("Drosophila melanogaster", G_drosophila_binary,
#                          A_drosophila_binary)
# Drosophila melanogaster
# Number of nodes N = 21733
# Number of edges M = 2872500
# <k_in> = <k_out>= 162.01918741084987
# k_in_max = 5124
# k_out_max = 2723


A_mouse_meso = np.loadtxt("data/ABA_weight_mouse.txt")  # > 0
G_mouse_meso = nx.from_numpy_array(A_mouse_meso)
print_network_properties("Mus musculus", G_mouse_meso, A_mouse_meso)
# Mus musculus
# Number of nodes N = 213
# Number of edges M = 1726
# <k_in> = <k_out>= 4.323531093331828
# k_in_max = 42
# k_out_max = 57


plt.figure(figsize=(8, 6))
markersize = 15
# plt.subplot(221)
# ax = plt.gca()
# plot_spectrum(ax, A_star, 1, vals=[],
#               axvline_color="#2171b5",
#               bar_color=reduced_first_community_color,
#               normed=False, xlabel="$\\lambda$",
#               ylabel="$\\rho(\\lambda)$",
#               label_fontsize=13, nbins=30, labelsize=13)
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8])

plt.subplot(221)
ax = plt.gca()

plot_complex_spectrum(A_ciona, 1,
                      color=reduced_first_community_color,
                      markersize=markersize,
                      xlabel="Re[$\\lambda$]",
                      ylabel="Im[$\\lambda$]",
                      label_fontsize=13, labelsize=13)

# plt.ylim([0, 0.15])
# plt.yticks([0, 0.05, 0.1, 0.15])


plt.subplot(222)
# ax = plt.gca()
# plot_complex_spectrum(ax, A_celegans, 1, vals=[],
#                       axvline_color="#2171b5",
#                       bar_color=reduced_second_community_color,
#                       normed=False, xlabel="$\\lambda$",
#                       ylabel="",
#                       label_fontsize=13, nbins=60, labelsize=13)
# plt.xticks([-10, 0, 10, 20])
# plt.yticks([0, 0.05, 0.1])
# plt.ylim([0, 0.101])
plot_complex_spectrum(A_celegans, 1,
                      color=reduced_second_community_color,
                      markersize=markersize,
                      xlabel="Re[$\\lambda$]",
                      ylabel="Im[$\\lambda$]",
                      label_fontsize=13, labelsize=13)


plt.subplot(223)
plot_complex_spectrum(A_drosophila_binary, 1,
                      color=reduced_third_community_color,
                      markersize=markersize,
                      xlabel="Re[$\\lambda$]",
                      ylabel="Im[$\\lambda$]",
                      label_fontsize=13, labelsize=13,
                      sparse=True, nb_eigen=1000)
# 67 minutes, for nb eigen=1000

plt.subplot(224)
# ax = plt.gca()
# plot_complex_spectrum(ax, A_mouse_meso, 1, vals=[],
#                       axvline_color="#2171b5",
#                       bar_color=reduced_first_community_color,
#                       normed=False, xlabel="$\\lambda$",
#                       ylabel="$\\rho(\\lambda)$",
#                       label_fontsize=13, nbins=60, labelsize=13)
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8])
plot_complex_spectrum(A_mouse_meso, 1,
                      color=reduced_fourth_community_color,
                      markersize=markersize,
                      xlabel="Re[$\\lambda$]",
                      ylabel="Im[$\\lambda$]",
                      label_fontsize=13, labelsize=13)

plt.tight_layout()

plt.show()
# """


# Spectre des valeurs propres réelles
"""
A_mouse_meso = np.loadtxt("data/ABA_weight_mouse.txt") > 0

G_mouse = nx.Graph()
edges = nx.read_edgelist('data/bn-mouse_visual-cortex_2.edges')
G_mouse.add_edges_from(edges.edges())
A_mouse = nx.to_numpy_array(G_mouse)

A_celegans = np.load("data/C_Elegans.npy")
G = nx.from_numpy_array(A_celegans)

A_from_xlsx = pd.read_excel('data/ciona_intestinalis_lavaire_elife'
                            '-16962-fig16-data1-v1_modified.xlsx').values
A_ciona_nan = np.array(A_from_xlsx[0:, 1:])
A_ciona = np.array(A_ciona_nan, dtype=float)
where_are_NaNs = np.isnan(A_ciona)
A_ciona[where_are_NaNs] = 0
A_ciona = A_ciona > 0

# print(np.sort(np.ndarray.flatten(A_ciona)), np.max(A_ciona))

# print(min(i for i in np.ndarray.flatten(A_ciona) if i > 0))
plt.figure(figsize=(6, 6))

yticks = [0, 0.1, 0.2, 0.3]
ylim_max = 0.35

plt.subplot(221)
ax = plt.gca()
plot_spectrum(ax, A_ciona, 1, vals=[],
              axvline_color="#2171b5",
              bar_color=reduced_first_community_color,
              normed=False, xlabel="$\\lambda$",
              ylabel="$\\rho(\\lambda)$",
              label_fontsize=13, nbins=20, labelsize=13)
plt.yticks(yticks)
plt.ylim([0, ylim_max])

plt.subplot(222)
ax = plt.gca()
plot_spectrum(ax, A_celegans, 1, vals=[],
              axvline_color="#2171b5",
              bar_color=reduced_second_community_color,
              normed=False, xlabel="$\\lambda$",
              ylabel="",
              label_fontsize=13, nbins=20, labelsize=13)
plt.xticks([-10, 0, 10, 20])
plt.yticks(yticks)
plt.ylim([0, ylim_max])


plt.subplot(223)
ax = plt.gca()
plot_spectrum(ax, A_mouse_meso, 1, vals=[],
              axvline_color="#2171b5",
              bar_color=reduced_fourth_community_color,
              normed=False, xlabel="$\\lambda$",
              ylabel="$\\rho(\\lambda)$",
              label_fontsize=13, nbins=20, labelsize=13)
plt.xticks([-3, 0, 3, 6, 9])
plt.yticks(yticks)
plt.ylim([0, ylim_max])

plt.subplot(224)
ax = plt.gca()
plot_spectrum(ax, A_mouse, 1, vals=[],
              axvline_color="#2171b5",
              bar_color=reduced_third_community_color,
              normed=False, xlabel="$\\lambda$",
              ylabel="",
              label_fontsize=13, nbins=60, labelsize=13)
plt.ylim([0, 0.05])
plt.yticks([0, 0.02, 0.04])

plt.show()

"""
