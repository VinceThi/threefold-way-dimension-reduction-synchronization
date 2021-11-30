import dynamicalab.drawing as draw
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

first_community_color = "#2171b5"
second_community_color = "#f16913"
third_community_color = "#6a51a3"
fourth_community_color = "#ef3b2c"
mu = 0
edge_color = "gray"
edge_width = 1  # 1.5
edge_alpha = 0.3  # 0.6
use_edge_weigth = False
node_width = 1  # 1.5
node_size = 20.0  # 50
reduced_node_size = 100.0
node_border_color = "#404040"
node_alpha = 1.0
arrow_scale = 0
loop_radius = 0  # 0

path_str = "C:/Users/thivi/Documents/GitHub/network-synch/" \
           "synch_predictions/graphs/binary_matrices/"

""" Heterogeneous network """
# A = np.array(np.loadtxt(path_str+"dMat_bin_h=1.00_rho=-0.90.txt"))
# A = np.array(np.loadtxt(path_str+"dMat_bin_h=1.00_rho=-0.90_sparse.txt"))

""" S1 network """
A = np.array(np.loadtxt(path_str+"dMat_bin_example3.1.txt"))


plt.matshow(A, aspect="auto")
plt.show()

N = len(A[:, 0])

plt.figure(figsize=(4, 4))
ax = plt.subplot(111)
G_ts = nx.from_numpy_matrix(A)
# pos = nx.layout.kamada_kawai_layout(G_ts)
# pos = nx.layout.spectral_layout(G_ts)
# pos = nx.spring_layout(G_ts,  pos=pos, iterations=1000)
pos = nx.circular_layout(G_ts)
node_colors = N*[first_community_color]
# ["#deebf7"] + (n1-1) * [first_community_color]
#  + ["#fee6ce"] + (n2-1) * [second_community_color]
draw.draw_networks(G_ts, pos, ax,
                   mu=mu,
                   edge_color=edge_color,
                   edge_width=edge_width,
                   edge_alpha=edge_alpha,
                   use_edge_weigth=use_edge_weigth,
                   node_width=node_width,
                   node_size=node_size,
                   node_border_color=node_border_color,
                   node_color=node_colors,
                   node_alpha=node_alpha,
                   arrow_scale=arrow_scale,
                   loop_radius=loop_radius)
plt.show()
