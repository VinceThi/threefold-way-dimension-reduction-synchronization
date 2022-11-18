from graphs.get_reduction_matrix_and_characteristics import *
from plots.plot_principal_component_analysis import *
import json
# import umap
from sklearn.decomposition import PCA


# --------------------------- Get data  ---------------------------------------
""" Two-triangles, n = 2 """
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_20_21h22min04sec_CVM_two_triangles_2D_dictionary' \
#                 '_dominant_V_A.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_15h12min47sec_CVM_two_triangles_2D_' \
#                 'dictionary_V_K_all_nodes.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_18h41min12sec_CVM_two_triangles_2D_dictionary' \
#                 '_V_K_one_less_node.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_18h51min35sec_CVM_two_triangles_2D_dictionary' \
#                 '_V_K_perturbed.json'
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
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_06_02h29min09sec_CVM_dictionary_two_triangles" \
#                 "_2D_not_aligned.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_07_16h48min16sec_CVM_dictionary_two_triangles" \
#                 "_2D_not_aligned_2.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_06_17h02min18sec_CVM_dictionary_two_triangles" \
#                 "_2D_multiple_inits.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_27_16h47min12sec_CVM_dictionary_two_triangles" \
#                 "_2D_V_K_all_nodes.json"


""" Two-triangles, n = 3 """
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_20_21h22min31sec_CVM_two_triangles' \
#                 '_3D_dictionary_V0_V1_V3.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_19h58min26sec_CVM_two_triangles_3D' \
#                 '_dictionary_V0_V1_V3.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_02_01_02h10min58sec_CVM_dictionary_two_triangles_3D' \
#                 '_snmf_and_onmf.json'
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_06_01h58min10sec_CVM_dictionary_two_triangles" \
#                 "_3D_snmf_onmf_multiple_inits.json"
#
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_07_17h15min14sec_CVM_dictionary_two_triangles" \
#                 "_3D_multiple_inits.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_28_13h51min30sec_CVM_dictionary_two_triangles" \
#                 "_3D_V_K_all_nodes_V_A_dominant_veps.json"
CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
                "synch_predictions/graphs/two_triangles/CVM_data/" \
                "2020_02_28_18h31min13sec_CVM_dictionary_two_triangles_3D" \
                "_V_K_all_nodes_V_A_V0_V1_V3.json"


""" Small SBM, n = 2 """
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


""" Small bibartite, n = 3 """
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
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/SBM/CVM_data/' \
#                 '2020_02_05_12h23min47sec_CVM_dictionary_bipartite' \
#                 '_2D_V_W_V_K_approx.json'
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/SBM/CVM_data/" \
#                 "2020_02_09_15h40min53sec_CVM_dictionary_bipartite" \
#                 "_2D_bipartite_pout_0_2.json"


# ---------------------------- PCA --------------------------------------------

""" Multiple targets"""
# """
with open(f'{CVM_dict_path}') as json_data:
    CVM_dictionary = json.load(json_data)

# Get data matrix
X, targets_possibilities = \
    get_matrix_of_flatten_reduction_matrices_for_all_targets(CVM_dictionary)

# Get components

M_W = CVM_dictionary["M_W"]
n, N = np.shape(M_W)
n_components = 3
svd_solver = 'auto'
pca = PCA(n_components=n_components, svd_solver=svd_solver)
pca.fit(X)
explained_variance_ratio = np.round(pca.explained_variance_ratio_, 2)
principal_component_matrix = pca.components_
X_transform = pca.transform(X)

# print(X.shape, principal_component_matrix.shape)

# Plot
plot_three_PCA_and_components(X_transform, principal_component_matrix, n, N,
                              explained_variance_ratio)
# """

""" Multiple realizations """
"""
plot_nmf_errors = 0

# Get data matrix
realizations_dictionary_str = "SBM_theta"
realizations_dictionary_path = \
    get_realizations_dictionary_absolute_path(realizations_dictionary_str)
with open(f'C:/Users/thivi/Documents/GitHub/network-synch/'                 
          f'synch_predictions/simulations/data/'                            
          f'synchro_transitions_multiple_realizations/'                     
          f'{realizations_dictionary_path}.json') as json_data:
    realizations_dictionary = json.load(json_data)
X = get_matrix_of_flatten_reduction_matrices_for_all_realizations(
    realizations_dictionary)

# Get components
n, N = np.shape(realizations_dictionary["M_realizations"][0])
n_components = 3
svd_solver = 'auto'
pca = PCA(n_components=n_components, svd_solver=svd_solver)
pca.fit(X)
explained_variance_ratio = np.round(pca.explained_variance_ratio_, 4)
principal_component_matrix = pca.components_
X_transform = pca.transform(X)

for M in realizations_dictionary["M_realizations"]:
    M = np.array(M)
    if M[0][0] < M[0][-1]:
        M[[0, 1]] = M[[1, 0]]
    plt.matshow(M, aspect="auto")
    plt.show()

if plot_nmf_errors:
    snmf_ferr_realizations = np.array(
        realizations_dictionary["snmf_ferr_realizations"])
    onmf_ferr_realizations = np.array(
        realizations_dictionary["onmf_ferr_realizations"])
    onmf_oerr_realizations = np.array(
        realizations_dictionary["onmf_oerr_realizations"])

    plt.figure(figsize=(6, 3))
    plt.scatter(np.arange(len(snmf_ferr_realizations)), snmf_ferr_realizations,
                label="snmf ferr")
    plt.scatter(np.arange(len(onmf_ferr_realizations)), onmf_ferr_realizations,
                label="onmf ferr")
    plt.scatter(np.arange(len(onmf_oerr_realizations)), onmf_oerr_realizations,
                label="snmf oerr")
    plt.legend(loc="best")
    plt.xlabel("Number of realizations")
    plt.tight_layout()
    plt.show()

# Plot
plot_three_PCA_and_components_realizations(
    X_transform, principal_component_matrix, n, N, explained_variance_ratio)
"""

# ---------------------------- UMAP -------------------------------------------
"""
reducer = umap.UMAP()

embedding = reducer.fit_transform(X.T).T

print(embedding.shape)

marker = itertools.cycle(('*', '^', 'P', 'o', 's'))
markersize = itertools.cycle((140, 150, 150, 200, 200))
zorder = itertools.cycle((13, 10, 7, 4, 1))
color = 5*[c1] + 5*[c2] + 5*[c3]

plt.figure(figsize=(4, 5))
j = 0
for i in range(len(targets_possibilities)):
    if not i % 5 and i:
        j += 1
    plt.scatter(embedding[0][i],
                embedding[1][i],
                zorder=next(zorder)+j,
                s=next(markersize),
                color=color[i],
                marker=next(marker),
                label=targets_possibilities[i],
                edgecolors='#525252')
# plt.vlines(0, ylim[0], ylim[1], linewidth=1, color='k')
# plt.plot(np.linspace(xlim[0], xlim[1], 2), [0, 0], linewidth=1, color='k')
plt.ylabel(f"Embedding 2",
           fontsize=fontsize)
plt.xlabel(f"Embedding 1",
           fontsize=fontsize)
plt.legend(bbox_to_anchor=(1.15, -0.2),
           fontsize=fontsize_legend, ncol=3)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.tight_layout()
# plt.xticks(xticks)
# plt.yticks(yticks)
# plt.xlim(xlim)
# plt.ylim(ylim)
plt.show()
"""
