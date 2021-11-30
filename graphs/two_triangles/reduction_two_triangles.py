from synch_predictions.graphs.get_reduction_matrix_and_characteristics import *
from synch_predictions.plots.plot_spectrum import *
from synch_predictions.graphs.special_graphs import \
    two_triangles_graph_adjacency_matrix
from numpy.linalg import pinv

A = two_triangles_graph_adjacency_matrix()
K = np.diag(np.sum(A, axis=1))
W = np.diag(np.array([0.2, 0.2, 0.2, -0.2, -0.2, -0.2]))


vapW = np.linalg.eig(W)[0]
vapK = np.linalg.eig(K)[0]
vapvep = np.linalg.eig(A)


print("spec(W) = ", vapW, "\n",
      "spec(K) = ", vapK, "\n",
      "spec(A) = ", vapvep[0], "\n \n \n")

M_omega = np.array([[1/3, 1/3, 1/3, 0, 0, 0],
                    [0, 0, 0, 1/3,  1/3, 1/3]])

# ------------------------- One target W --------------------------------------
print("Frequency reduction: One target W \n \n",
      "redW0 = M_omega W M_omega^+ =\n", M_omega@W@pinv(M_omega),
      "\n \n spec(redW) = ", np.linalg.eig(M_omega@W@pinv(M_omega))[0],
      "\n \n np.sqrt(||M_omega W - M_omega W M_omega^+M_omega ||^2) =",
      rmse(M_omega @ W, M_omega @ W @ pinv(M_omega) @ M_omega),
      "\n \n redK0 = M_omega K M_omega^+ =\n", M_omega@K@pinv(M_omega),
      "\n \n np.sqrt(||M_omega K - M_omega K M_omega^+M_omega ||^2) =",
      rmse(M_omega @ K, M_omega @ K @ pinv(M_omega) @ M_omega),
      "\n \n spec(redK) = ", np.linalg.eig(M_omega@K@pinv(M_omega))[0],
      "\n \n redA = M_omega A M_omega^+ =\n", M_omega@A@pinv(M_omega),
      "\n \n spec(redA) = ", np.linalg.eig(M_omega@A@pinv(M_omega))[0],
      "\n \n np.sqrt(||M_omega A - M_omega A M_omega^+M_omega ||^2) =",
      rmse(M_omega @ A, M_omega @ A @ pinv(M_omega) @ M_omega),
      "\n \n \n")

M_k = np.array([[1/2, 1/2, 0, 0, 0, 0],
                [0, 0, 1/2, 1/2,  0, 0],
                [0, 0, 0, 0, 1/2, 1/2]])

# ------------------------- One target K --------------------------------------
print("Degree reduction: One target K \n \n",
      "redW = M_k W M_k^+ = \n", M_k@W@pinv(M_k),
      "\n \n spec(redW) = ", np.linalg.eig(M_k@W@pinv(M_k))[0],
      "\n \n np.sqrt(||M_k W - M_k W M_k^+M_k ||^2) =",
      rmse(M_k @ W, M_k @ W @ pinv(M_k) @ M_k),
      "\n \n redK = M_k K M_k^+ =\n", M_k@K@pinv(M_k),
      "\n \n spec(redK) = ", np.linalg.eig(M_k@K@pinv(M_k))[0],
      "\n \n np.sqrt(||M_k K - M_k K M_k^+M_k ||^2) =",
      rmse(M_k @ K, M_k @ K @ pinv(M_k) @ M_k),
      "\n \n redA0 = M_k A M_k^+ =\n", M_k@A@pinv(M_k),
      "\n \n spec(redA) = ", np.linalg.eig(M_k@A@pinv(M_k))[0],
      "\n \n np.sqrt(||M_k A - M_k A M_k^+M_k ||^2) =",
      rmse(M_k @ A, M_k @ A @ pinv(M_k) @ M_k),
      "\n \n \n")


# ----------------------- Two target A then W ---------------------------------

V0 = vapvep[1][:, 0]
V1 = vapvep[1][:, 1]
V2 = vapvep[1][:, 2]
V3 = vapvep[1][:, 3]
V4 = vapvep[1][:, 4]
V5 = vapvep[1][:, 5]

V_2D = np.array([V0, V1])

print("V^+ V = ", np.sum(np.dot(pinv(V_2D), V_2D), axis=1), "\n\n\n")

C_2D = np.dot(M_omega, pinv(V_2D))

CV_2D = np.dot(C_2D, V_2D)

M_2D = (CV_2D.T / (np.sum(CV_2D, axis=1))).T   # CV_2D

# print("alllo", np.sum(V_2D, axis=1), "\n\n\n")
# print("alllo", V_2D, "\n\n\n")
# print("alllo", np.sum(pinv(V_2D), axis=1), "\n\n\n")
# print("alllo", np.sum(M_2D, axis=1), "\n\n\n")

print("Spectral reduction 2D: Two target A then W \n \n",
      "V_2D = ", V_2D, "\n \n V_2D^+ = ", pinv(V_2D), " = V^T \n",
      "\n Vérif VEPs:\n", np.dot(np.dot(A, V_2D[0].T), V_2D[0]) /
      np.dot(V_2D[0].T, V_2D[0]),
      "\n", np.dot(np.dot(A, V_2D[1].T), V_2D[1])/np.dot(V_2D[1].T, V_2D[1]),
      "\n \n C_2D = M_k V^+ =\n", C_2D,
      "\n \n C_2D^T C_2D =\n", np.dot(C_2D.T, C_2D),
      "\n \n C_2D^+ C_2D =\n", np.dot(pinv(C_2D), C_2D),
      "\n \n M_2D= ", M_2D,
      "\n \n L1 error = np.mean(|M_2D - M_W|) = ", np.mean(np.abs(M_2D -
                                                                  M_omega)),
      "\n \n M_2D M_2D^+ =\n", np.dot(M_2D, pinv(M_2D)),
      "\n \n M_2D^+ M_2D =\n", np.dot(pinv(M_2D), M_2D),
      "\n \n redW = M_2D W M_2D^+ =\n", M_2D@W@pinv(M_2D),
      "\n \n spec(redW) = ", np.linalg.eig(M_2D@W@pinv(M_2D))[0],
      "\n \n np.sqrt(||M_2D W - M_2D W M_2D^+M_2D ||^2) =",
      rmse(M_2D @ W, M_2D @ W @ pinv(M_2D) @ M_2D),
      "\n \n redK = M_2D K M_2D^+ =\n", M_2D@K@pinv(M_2D),
      "\n \n spec(redK) = ", np.linalg.eig(M_2D@K@pinv(M_2D))[0],
      "\n \n np.sqrt(||M K - M K M^+M ||^2) =",
      rmse(M_2D @ K, M_2D @ K @ pinv(M_2D) @ M_2D),
      "\n \n redA = M_2D A M_2D^+ =\n", M_2D@A@pinv(M_2D),
      "\n \n spec(redA) = ", np.linalg.eig(M_2D@A@pinv(M_2D))[0],
      "\n \n np.sqrt(||M A - M A M^+M ||^2) =",
      rmse(M_2D @ A, M_2D @ A @ pinv(M_2D) @ M_2D),
      "\n \n \n")


# ----------------------- Two target A then K ---------------------------------
V = np.array([V0, V1, V3])

C = np.dot(M_k, pinv(V))

CV = np.dot(C, V)

M = (CV.T / (np.sum(CV, axis=1))).T

print("Spectral reduction 3D \n \n",
      "V = ", V, "\n \n V^+ = ", pinv(V), " = V^T \n",
      "\n Vérif VEPs:\n", np.dot(np.dot(A, V[0].T), V[0])/np.dot(V[0].T, V[0]),
      "\n", np.dot(np.dot(A, V[1].T), V[1])/np.dot(V[1].T, V[1]),
      "\n", np.dot(np.dot(A, V[2].T), V[2])/np.dot(V[2].T, V[2]),
      "\n \n C = M_k V^+ =\n", C,
      "\n \n C^T C =\n", np.dot(C.T, C),
      "\n \n C^+ C =\n", np.dot(pinv(C), C),
      "\n \n M = ", M,
      "\n \n L1 error = np.mean(|M - M_k|) = ", np.mean(np.abs(M - M_k)),
      "\n \n M M^+ =\n", np.dot(M, pinv(M)),
      "\n \n M^+ M =\n", np.dot(pinv(M), M),
      "\n \n redW = M W M^+ =\n", M@W@pinv(M),
      "\n \n spec(redW) = ", np.linalg.eig(M@W@pinv(M))[0],
      "\n \n np.sqrt(||M W - M W M^+M ||^2) =",
      rmse(M @ W, M @ W @ pinv(M) @ M),
      "\n \n redK = M K M^+ =\n", M@K@pinv(M),
      "\n \n spec(redK) = ", np.linalg.eig(M@K@pinv(M))[0],
      "\n \n np.sqrt(||M K - M K M^+M ||^2) =",
      rmse(M @ K, M @ K @ pinv(M) @ M),
      "\n \n redA = M A M^+ =\n", M@A@pinv(M),
      "\n \n spec(redA) = ", np.linalg.eig(M@A@pinv(M))[0],
      "\n \n np.sqrt(||M A - M A M^+M ||^2) =",
      rmse(M @ A, M @ A @ pinv(M) @ M))


# ----------------------- Two target W then A ---------------------------------


V_W = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
                [0, 0, 0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])
VA_VWp_VW = V_2D@pinv(V_W)@V_W
M_WA = VA_VWp_VW  # .T # / (np.sum(VA_VWp_VW, axis=1))).T

# print(V_2D@pinv(V_W)@V_W )

print("Reduction 2D: Two target W then A \n \n",
      "\n \n M_WA= ", M_WA,
      "\n \n M_WA M_WA^+ =\n", np.dot(M_WA, pinv(M_WA)),
      "\n \n M_WA^+ M_WA =\n", np.dot(pinv(M_WA), M_WA),
      "\n \n redW = M_WA W M_WA^+ =\n", M_WA@W@pinv(M_WA),
      "\n \n spec(redW) = ", np.linalg.eig(M_WA@W@pinv(M_WA))[0],
      "\n \n np.sqrt(||M_WA W - M_WA W M_WA^+M_WA ||^2) =",
      rmse(M_WA @ W, M_WA @ W @ pinv(M_WA) @ M_WA),
      "\n \n redK = M_WA K M_WA^+ =\n", M_WA@K@pinv(M_WA),
      "\n \n spec(redK) = ", np.linalg.eig(M_WA@K@pinv(M_WA))[0],
      "\n \n np.sqrt(||M K - M K M^+M ||^2) =",
      rmse(M_WA @ K, M_WA @ K @ pinv(M_WA) @ M_WA),
      "\n \n redA = M_WA A M_WA^+ =\n", M_WA@A@pinv(M_WA),
      "\n \n spec(redA) = ", np.linalg.eig(M_WA@A@pinv(M_WA))[0],
      "\n \n np.sqrt(||M A - M A M^+M ||^2) =",
      rmse(M_WA @ A, M_WA @ A @ pinv(M_WA) @ M_WA),
      "\n \n \n")

# ---------------------- Two target K then W ----------------------------------

V_W_3D = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]])

V_K_3D = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0],
                   [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
                   [0, 0, 0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)]])

VW_VKp_VK = V_W_3D @ pinv(V_K_3D) @ V_K_3D
M_KW = (VW_VKp_VK.T / (np.sum(VW_VKp_VK, axis=1))).T

print("Degree reduction 3D: Two target K then W \n \n",
      "\n \n M_KW= ", M_KW,
      "\n \n M_KW M_KW^+ =\n", np.dot(M_KW, pinv(M_KW)),
      "\n \n M_KW^+ M_KW =\n", np.dot(pinv(M_KW), M_KW),
      "\n \n redW = M_KW W M_KW^+ =\n", M_KW@W@pinv(M_KW),
      "\n \n spec(redW) = ", np.linalg.eig(M_KW@W@pinv(M_KW))[0],
      "\n \n np.sqrt(||M_KW W - M_KW W M_KW^+M_KW ||^2) =",
      rmse(M_KW @ W, M_KW @ W @ pinv(M_KW) @ M_KW),
      "\n \n redK = M_KW K M_KW^+ =\n", M_KW@K@pinv(M_KW),
      "\n \n spec(redK) = ", np.linalg.eig(M_KW@K@pinv(M_KW))[0],
      "\n \n np.sqrt(||M_KW K - M_KW K M_KW^+M_KW ||^2) =",
      rmse(M_KW @ K, M_KW @ K @ pinv(M_KW) @ M_KW),
      "\n \n redA = M_KW A M_KW^+ =\n", M_KW@A@pinv(M_KW),
      "\n \n spec(redA) = ", np.linalg.eig(M_KW@A@pinv(M_KW))[0],
      "\n \n np.sqrt(||M_KW A - M_KW A M_KW^+M_KW ||^2) =",
      rmse(M_KW @ A, M_KW @ A @ pinv(M_KW) @ M_KW))


# def coefficent_constraints(c1, c2, c3, V):
#     c4 = -c1*c3/c2
#     v1 = V[0]
#     v2 = V[1]
#     return [c1**2 + c2**2 - 1, c3**2 + c1**2*c3**2/(c2**2) - 1,
#
# Tests for curiosity
# sizes = [20, 10]
# n1, n2 = sizes
# pq = [[0, 0.5], [0.5, 0]]
# A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
# W = np.diag(np.concatenate([0.1*np.ones(n1), 0.3*np.ones(n2)]))
# # W = 1j*np.diag([-0.1, -0.1, -0.1, 0.1, 0.1, 0.1])
# K = np.diag(np.sum(A, axis=1))
# B = 1j*W + K + A
#
# print(B, np.linalg.eig(B)[0])
# fig = plt.figure()
# plt.plot(np.linalg.eig(B)[1][:, 0:2])
# # fig.colorbar(cax)
# plt.show()
#
# plt.figure(figsize=(3, 3))
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
# ax1 = plt.subplot(111)
# plot_spectrum(ax1, B, 1000, vals=[],
#               axvline_color="#2171b5", bar_color="#2171b5",
#               normed=True, xlabel="$\\lambda$", ylabel="$\\rho_A(\\lambda)$",
#               label_fontsize=13, nbins=24, labelsize=13)
#
# plt.show()
