from synch_predictions.plots.plot_distributions_and_sequences import *
import numpy as np
import json

first_community_color = "#2171b5"
second_community_color = "#f16913"
reduced_first_community_color = "#9ecae1"
reduced_second_community_color = "#fdd0a2"
reduced_third_community_color = "#a1d99b"
total_color = "#525252"
fontsize = 12
inset_fontsize = 9
fontsize_legend = 12
labelsize = 12
inset_labelsize = 9
linewidth = 2
labelpad = 10
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

""" Two-triangles, n = 2 """
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_20_21h22min04sec_CVM_two_triangles_2D_dictionary' \
#                 '_dominant_V_A.json'
# sigma_array = np.linspace(0.01, 8, 500)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_15h12min47sec_CVM_two_triangles_2D_' \
#                 'dictionary_V_K_all_nodes.json'
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_18h41min12sec_CVM_two_triangles_2D_dictionary' \
#                 '_V_K_one_less_node.json'
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_18h51min35sec_CVM_two_triangles_2D_dictionary' \
#                 '_V_K_perturbed.json'
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_02_01_02h10min15sec_CVM_dictionary_two_triangles' \
#                 '_2D_snmf_and_onmf.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_02_04_20h01min19sec_CVM_dictionary_two_triangles' \
#                 '_2D_multiple_inits_V_K_perturbed.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_02_04_23h56min37sec_CVM_dictionary_two_triangles' \
#                 '_2D_multiple_inits_V_K_one_neglected_node.json'
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_06_01h54min15sec_CVM_dictionary_two_triangles" \
#                 "_2D_snmf_onmf_multiple_inits.json"
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_06_02h29min09sec_CVM_dictionary_two_triangles" \
#                 "_2D_not_aligned.json"
CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
                "synch_predictions/graphs/two_triangles/CVM_data/" \
                "2020_02_06_17h02min18sec_CVM_dictionary_two_triangles" \
                "_2D_multiple_inits.json"
# sigma_array = np.linspace(0.1, 4, 100)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_07_16h48min16sec_CVM_dictionary_two_triangles" \
#                 "_2D_not_aligned_2.json"
# sigma_array = np.linspace(0.1, 4, 100)

""" Two-triangles, n = 3 """
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_20_21h22min31sec_CVM_two_triangles' \
#                 '_3D_dictionary_V0_V1_V3.json'
# sigma_array = np.linspace(0.01, 8, 500)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_19h58min26sec_CVM_two_triangles_3D' \
#                 '_dictionary_V0_V1_V3.json'
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_02_01_02h10min58sec_CVM_dictionary_two_triangles_3D' \
#                 '_snmf_and_onmf.json'
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_06_01h58min10sec_CVM_dictionary_two_triangles" \
#                 "_3D_snmf_onmf_multiple_inits.json"
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_07_17h15min14sec_CVM_dictionary_two_triangles" \
#                 "_3D_multiple_inits.json"
# sigma_array = np.linspace(0.1, 4, 100)

""" Small bipartite, n = 2 """
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_21_16h33min34sec_CVM_small_bipartite_2D'  \
#                 '_dictionary_V_K_neglects_node_1.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_21_22h05min41sec_CVM_small_bipartite_2D' \
#                 '_dictionary_dominant_veps.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_23_14h41min07sec_CVM_small_bipartite_2D' \
#                 '_dictionary_deg_freq.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_23_15h11min37sec_CVM_small_bipartite_2D' \
#                 '_dictionary_other_deg_freq.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_29_23h36min49sec_CVM_dictionary_' \
#                 'small_bipartite_2D_dom_eig.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_02_01_00h19min29sec_CVM_dictionary_small_bipartite' \
#                 '_2D_snmf_and_onmf.json'

""" Small bipartite, n = 3 """
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_21_21h49min46sec_CVM_small_bipartite_3D' \
#                 '_dictionary_with_eigA_0.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_21_21h56min08sec_CVM_small_bipartite_3D' \
#                 '_dictionary_dominant_veps.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_23_15h06min19sec_CVM_small_bipartite_3D' \
#                 '_dictionary_deg_freq.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_23_15h12min31sec_CVM_small_bipartite_3D' \
#                 '_dictionary_other_deg_freq.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_30_00h04min39sec_CVM_dictionary_small_bipartite' \
#                 '_3D_V0_V2_V1_dom_eig.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_02_01_00h33min30sec_CVM_dictionary_small_bipartite' \
#                 '_3D_snmf_and_onmf.json'

""" Bipartite, n = 2 """
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/SBM/CVM_data/" \
#                 "2020_02_09_15h40min53sec_CVM_dictionary_bipartite" \
#                 "_2D_bipartite_pout_0_2.json"

with open(f'{CVM_dict_path}') as json_data:
    CVM_dict = json.load(json_data)
W = np.array(CVM_dict["W"])
K = np.array(CVM_dict["K"])
A = np.array(CVM_dict["A"])

frequency_sequence = np.diag(W)
degree_sequence = np.diag(K)


plt.figure(figsize=(3, 3.5))

ax = plt.subplot(311)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# plot_distribution(ax, frequency_sequence,
#                   width=0.0001,
#                   color=reduced_first_community_color,
#                   xy_log_scale=False)
plt.plot(np.linspace(1, 6, 6), np.zeros(6), "k", linewidth=0.75)
plot_sequence(ax, frequency_sequence, xlabel="", ylabel="$\\omega_j$",
              width=0.5, yticks=(-0.2, 0, 0.2), labelsize=labelsize,
              color=reduced_first_community_color, fontsize=fontsize,
              xy_log_scale=False)


ax = plt.subplot(312)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plot_distribution(ax, degree_sequence,
#                   width=0.5,
#                   color=reduced_second_community_color,
#                   xy_log_scale=False)
plot_sequence(ax, degree_sequence, xlabel="", ylabel="$k_j$",
              width=0.5, yticks=(0, 1, 2, 3), labelpad=10, fontsize=fontsize,
              color=reduced_second_community_color, labelsize=labelsize,
              xy_log_scale=False)


ax = plt.subplot(313)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plot_distribution(ax, np.linalg.eig(A)[1][:, 0],
#                   width=0.0001,
#                   color=reduced_third_community_color,
#                   xy_log_scale=False)
plot_sequence(ax, -np.linalg.eig(A)[1][:, 0], xlabel="$j$",
              ylabel="$(\\mathbf{v}_D)_j$",  labelpad=20, fontsize=fontsize,
              width=0.5, yticks=(0, 0.5), labelsize=labelsize,
              color=reduced_third_community_color,
              xy_log_scale=False)
plt.show()

# Old
"""
plot_two_freq = 1                                 
plot_two_triangles_degrees = 0                    
plot_small_bipartite_degrees = 0                  
plot_small_bipartite_degrees_freq = 0             
plot_small_bipartite_degrees_freq_2 = 0           
nodes_label = np.arange(1, 7, 1)                  

if plot_two_freq:
    omega_array = 3 * [0.2] + 3 * [-0.2]
    fig = plt.figure(figsize=(3, 1.2))
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(np.linspace(0, 7, 6), np.zeros(6), "k", linewidth=0.5)
    plt.bar(nodes_label, omega_array, color=reduced_first_community_color)
    plt.ylim([-0.25, 0.25])
    plt.xlim([0.5, 6.5])
    # plt.xlabel("$j$", fontsize=fontsize)
    # ylab = plt.ylabel("$\\omega_j$", fontsize=fontsize)
    # ylab.set_rotation(0)
    # plt.xticks(nodes_label)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    plt.yticks([-0.2, 0,  0.2])
    plt.tick_params(
        axis='x',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        bottom=False,       # ticks along the bottom edge are off
        top=False,          # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    ax1.yaxis.set_ticks_position('left')
    # ax1.xaxis.set_ticks_position('bottom')
    # ax1.xaxis.set_ticks_position('bottom')
    plt.text(x=0.9, y=-0.11, s="$1$", fontsize=fontsize)
    plt.text(x=1.9, y=-0.11, s="$2$", fontsize=fontsize)
    plt.text(x=2.9, y=-0.11, s="$3$", fontsize=fontsize)
    plt.text(x=3.9, y=0.04, s="$4$", fontsize=fontsize)
    plt.text(x=4.9, y=0.04, s="$5$", fontsize=fontsize)
    plt.text(x=5.9, y=0.04, s="$6$", fontsize=fontsize)
    plt.text(x=6.7, y=-0.02, s="$j$", fontsize=fontsize)
    plt.text(x=0.2, y=0.3, s="$\\omega_j$", fontsize=fontsize)
    plt.tight_layout()
    plt.show()

elif plot_two_triangles_degrees:
    k_array = np.sum(two_triangles_graph_adjacency_matrix(), axis=1)
    fig = plt.figure(figsize=(3, 1.2))
    ax1 = plt.subplot(1, 1, 1)
    plt.bar(nodes_label, k_array, color=reduced_second_community_color)
    plt.xlim([0.5, 6.5])
    plt.text(x=6.7, y=-0.02, s="$j$")
    # ylab = plt.ylabel("$k_j$", fontsize=fontsize, labelpad=10)
    # ylab.set_rotation(0)
    plt.text(x=0.4, y=3.6, s="$k_j$", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.yticks([0, 1, 2, 3])
    plt.xticks(nodes_label)
    ax1.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.show()


elif plot_small_bipartite_degrees:
    k_array = np.sum(small_bipartite_graph_adjacency_matrix(), axis=1)
    fig = plt.figure(figsize=(3, 1.2))
    ax1 = plt.subplot(1, 1, 1)
    plt.bar(nodes_label, k_array, color=reduced_second_community_color)
    plt.xlim([0.5, 6.5])
    plt.text(x=6.7, y=-0.02, s="$j$")
    # ylab = plt.ylabel("$k_j$", fontsize=fontsize, labelpad=10)
    # ylab.set_rotation(0)
    plt.text(x=0.4, y=3.6, s="$k_j$", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.yticks([0, 1, 2, 3])
    plt.xticks(nodes_label)
    ax1.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.show()

elif plot_small_bipartite_degrees_freq:
    k_array = np.sum(small_bipartite_graph_adjacency_matrix(), axis=1)
    fig = plt.figure(figsize=(3, 1.2))
    ax1 = plt.subplot(1, 1, 1)
    plt.bar(nodes_label, k_array, color=reduced_second_community_color)
    plt.xlim([0.5, 6.5])
    plt.text(x=6.7, y=-0.02, s="$j$")
    # ylab = plt.ylabel("$k_j$\n$ = 10\:\\omega_j$", fontsize=fontsize,
    #                   labelpad=20)
    # ylab.set_rotation(0)
    plt.text(x=-0.2, y=3.6, s="$k_j= 10\:\\omega_j$", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.yticks([0, 1, 2, 3])
    plt.xticks(nodes_label)
    ax1.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.show()

elif plot_small_bipartite_degrees_freq_2:
    k_array = np.sum(small_bipartite_graph_adjacency_matrix(), axis=1)
    fig = plt.figure(figsize=(3, 1.2))
    ax1 = plt.subplot(1, 1, 1)
    plt.bar(nodes_label, k_array/10, color=reduced_first_community_color)
    plt.xlim([0.5, 6.5])
    plt.text(x=6.7, y=-0.02, s="$j$")
    # ylab = plt.ylabel("$k_j$\n$ = 10\:\\omega_j$", fontsize=fontsize,
    #                   labelpad=20)
    # ylab.set_rotation(0)
    plt.text(x=-0.2, y=0.36, s="$\\omega_j= k_j/10$", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.yticks([0, 0.1, 0.2, 0.3])
    plt.xticks(nodes_label)
    ax1.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.show()
"""
