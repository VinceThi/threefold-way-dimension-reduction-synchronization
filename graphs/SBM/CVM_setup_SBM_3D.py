from synch_predictions.graphs.get_reduction_matrix_and_characteristics \
    import *
from synch_predictions.plots.plot_distributions_and_sequences import *
from synch_predictions.graphs.graph_spectrum import get_eigenvectors_matrix

N1 = 300
N2 = 200
sizes = [N1, N2]
N = N1 + N2
affinity_matrix = [[0, 0.2],
                   [0.2, 0]]
G_SBM = nx.stochastic_block_model(sizes, affinity_matrix)
A = nx.to_numpy_array(G_SBM)
if affinity_matrix[0][0] == 0 and affinity_matrix[1][1] == 0:
    graph_str = "bipartite"
else:
    graph_str = "SBM"
parameters_dictionary = {"N1": N1,
                         "N2": N2,
                         "affinity_matrix": affinity_matrix,
                         "graph_str": graph_str}
k_sequence = get_degree_sequence(G_SBM)
K = np.diag(k_sequence)
# omega1 = 0.2
# omega2 = -N1/N2*omega1
# W = np.diag(N1*[omega1] + N2*[omega2])
omega1_random = np.random.normal(0.2, 0.01, N1)
omega2_random = -N1/N2*omega1_random[:N2]
# This choice of omega2 does not make the sum of omega1_random + omega2_random
# equal to zero, a better algorithm should be made for that
error_sum_omega1_omega2 = \
    np.sum(np.concatenate([omega1_random, omega2_random]))
term_to_be_added_to_omega2_random = error_sum_omega1_omega2/N2
omega2_random -= term_to_be_added_to_omega2_random*np.ones(N2)
W = np.diag(np.concatenate(([omega1_random,
                             omega2_random])))

k_sequence1 = k_sequence[:N1//2]
k_sequence2 = k_sequence[N1//2:N1]
k_sequence3 = k_sequence[N1:]
k_sequence1_norm = k_sequence1/np.sqrt(np.sum(k_sequence1**2))
k_sequence2_norm = k_sequence2/np.sqrt(np.sum(k_sequence2**2))
k_sequence3_norm = k_sequence3/np.sqrt(np.sum(k_sequence3**2))
V_K = np.block([[k_sequence1_norm, np.zeros(N1//2), np.zeros(N2)],
                [np.zeros(N1//2), k_sequence2_norm, np.zeros(N2)],
                [np.zeros(N1//2), np.zeros(N1//2), k_sequence3_norm]])

V_A = get_eigenvectors_matrix(A, 3)

omega_sequence1 = omega1_random[:N1//2]
omega_sequence2 = omega1_random[N1//2:]
omega_sequence3 = omega2_random
omega_sequence1_norm = omega_sequence1/np.sqrt(np.sum(omega_sequence1**2))
omega_sequence2_norm = omega_sequence2/np.sqrt(np.sum(omega_sequence2**2))
omega_sequence3_norm = omega_sequence3/np.sqrt(np.sum(omega_sequence3**2))
V_W = np.block([[omega_sequence1_norm, np.zeros(N1//2), np.zeros(N2)],
                [np.zeros(N1//2), omega_sequence2_norm, np.zeros(N2)],
                [np.zeros(N1//2), np.zeros(N1//2), omega_sequence3_norm]])


# print(f"\n \n V_W = {V_W}, \n \n V_K = {V_K}, \n \n V_A = {V_A}")

# plot_degree_distribution(G_SBM, width=0.5, color="#fdd0a2",
# xy_log_scale=False)

# plot_degree_sequence(G_SBM, width=0.5, color="#fdd0a2",
#                      xy_log_scale=False)

#  import matplotlib.pyplot as plt
#
# plt.figure(figsize=(8, 5))
#
# fig, axes = plt.subplots(nrows=3, ncols=1)
# axes[0].matshow(V_W, aspect="auto", cmap='jet')
# axes[1].matshow(V_K, aspect="auto", cmap='jet')
# axes[2].matshow(V_A, aspect="auto", cmap='jet')
# plt.tight_layout()
# plt.show()

if matrix_is_orthonormalized_VV_T(V_W) \
        and matrix_is_orthonormalized_VV_T(V_K)\
        and matrix_is_orthonormalized_VV_T(V_A):
    get_CVM_dictionary(W, K, A, V_W, V_K, V_A, graph_str,
                       parameters_dictionary, other_procedure=True)
else:
    raise ValueError("One or more eigenvector matrix is not orthonormal.")
