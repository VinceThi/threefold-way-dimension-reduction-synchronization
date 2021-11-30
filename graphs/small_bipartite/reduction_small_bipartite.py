from synch_predictions.graphs.get_reduction_matrix_and_characteristics import *
from synch_predictions.graphs.special_graphs import *
from numpy.linalg import pinv

A = small_bipartite_graph_adjacency_matrix()
K = np.diag(np.sum(A, axis=1))
W = np.diag(np.array([0.2, 0.2, 0.2, -0.2, -0.2, -0.2]))  # opposite freq
# W = K/10  # deg-freq

vapK = np.linalg.eig(K)[0]
vapW = np.linalg.eig(W)[0]
vapvep = np.linalg.eig(A)


print("spec(W) = ", vapW, "\n",
      "spec(K) = ", vapK, "\n",
      "spec(A) = ", vapvep[0], "\n \n \n")

M_omega = np.array([[1/3, 1/3, 1/3, 0, 0, 0],
                    [0, 0, 0, 1/3,  1/3, 1/3]])

print("Frequency reduction \n \n",
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

M_k = np.array([[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0,  0, 0],
                [0, 0, 1/4, 1/4, 1/4, 1/4]])

print("Degree reduction \n \n",
      "redW0 = M_k W M_k^+ = \n", M_k@W@pinv(M_k),
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

V0 = vapvep[1][:, 0]
V1 = vapvep[1][:, 1]
V2 = vapvep[1][:, 2]
V3 = vapvep[1][:, 3]
V4 = vapvep[1][:, 4]
V5 = vapvep[1][:, 5]
V = np.array([V0, V2])  # get_eigenvectors_matrix(A, 3)

C = np.dot(M_omega, pinv(V))

CV = np.dot(C, V)

M = (CV.T / (np.sum(CV, axis=1))).T

kappa = np.sum(np.dot(M, A), axis=1)  # equiv notation : alpha

print("Spectral reduction \n \n",
      "V = ", V, "\n \n V^+ = ", pinv(V), " = V^T \n",
      "\n Vérif VEPs:\n", np.dot(np.dot(A, V[0].T), V[0])/np.dot(V[0].T, V[0]),
      "\n", np.dot(np.dot(A, V[1].T), V[1])/np.dot(V[1].T, V[1]),
      # "\n", np.dot(np.dot(A, V[2].T), V[2])/np.dot(V[2].T, V[2]),
      "\n \n C = M_k V^+ =\n", C,
      "\n \n C^T C =\n", np.dot(C.T, C),
      "\n \n C^+ C =\n", np.dot(pinv(C), C),
      "\n \n M = ", M,
      "\n \n L1 error = np.mean(|M - M_omega|) = ", np.mean(np.abs(M-M_omega)),
      "\n \n M M^+ =\n", np.dot(M, pinv(M)),
      "\n \n M^+ M =\n", np.dot(pinv(M), M),
      "\n \n redW = M W M^+ =\n", M@W@pinv(M),
      "\n \n spec(redW) = ", np.linalg.eig(M@W@pinv(M))[0],
      "\n \n np.sqrt(||M W - M W M^+M ||^2) =",
      rmse(M @ W, M @ W @ pinv(M) @ M),
      "\n \n redK = M K M^+ =\n", M@K@pinv(M),
      "\n \n kappa = ", kappa,
      "\n \n spec(redK) = ", np.linalg.eig(M@K@pinv(M))[0],
      "\n \n np.sqrt(||M K - M K M^+M ||^2) =",
      rmse(M @ K, M @ K @ pinv(M) @ M),
      "\n \n redA = M A M^+ =\n", M@A@pinv(M),
      "\n \n spec(redA) = ", np.linalg.eig(M@A@pinv(M))[0],
      "\n \n np.sqrt(||M A - M A M^+M ||^2) =",
      rmse(M @ A, M @ A @ pinv(M) @ M))


V_3D = np.array([V0, V1, V2])
# This choice provokes a divergence in the integration of the reduced dynamics

C_3D = np.dot(M_k, pinv(V_3D))

CV_3D = np.dot(C_3D, V_3D)

M_3D = (CV_3D.T / (np.sum(CV_3D, axis=1))).T

print(" \n \n \n \nSpectral reduction 3D \n \n",
      "V_3D = ", V_3D, "\n \n V_3D^+ = ", pinv(V_3D), " = V_3D^T \n",
      "\n Vérif VEPs 3D:\n", np.dot(np.dot(A, V_3D[0].T), V_3D[0]) /
      np.dot(V_3D[0].T, V_3D[0]),
      "\n", np.dot(np.dot(A, V_3D[1].T), V_3D[1])/np.dot(V_3D[1].T, V_3D[1]),
      "\n", np.dot(np.dot(A, V_3D[2].T), V_3D[2])/np.dot(V_3D[2].T, V_3D[2]),
      # "\n", np.dot(np.dot(A, V[2].T), V[2])/np.dot(V[2].T, V[2]),
      "\n \n C_3D = M_k V^+ =\n", C_3D,
      "\n \n C_3D^T C_3D =\n", np.dot(C_3D.T, C_3D),
      "\n \n C_3D^+ C_3D =\n", np.dot(pinv(C_3D), C_3D),
      "\n \n M_3D = ", M_3D,
      "\n \n M_3D M_3D^+ =\n", np.dot(M_3D, pinv(M_3D)),
      "\n \n M_3D^+ M_3D =\n", np.dot(pinv(M_3D), M_3D),
      "\n \n redW_3D = M_3D W M_3D^+ =\n", M_3D@W@pinv(M_3D),
      "\n \n spec(redW_3D) = ", np.linalg.eig(M_3D@W@pinv(M_3D))[0],
      "\n \n np.sqrt(||M_3D W - M_3D W M_3D^+M_3D ||^2) =",
      rmse(M_3D @ W, M_3D @ W @ pinv(M_3D) @ M_3D),
      "\n \n redK_3D = M_3D K M_3D^+ =\n", M_3D@K@pinv(M_3D),
      "\n \n spec(redK_3D) = ", np.linalg.eig(M_3D@K@pinv(M_3D))[0],
      "\n \n np.sqrt(||M_3D K - M_3D K M_3D^+M_3D ||^2) =",
      rmse(M_3D @ K, M_3D @ K @ pinv(M_3D) @ M_3D),
      "\n \n redA_3D = M_3D A M_3D^+ =\n", M_3D@A@pinv(M_3D),
      "\n \n spec(redA_3D) = ", np.linalg.eig(M_3D@A@pinv(M_3D))[0],
      "\n \n np.sqrt(||M_3D A - M_3D A M_3D^+M_3D ||^2) =",
      rmse(M_3D @ A, M_3D @ A @ pinv(M_3D) @ M_3D))

print("\n (M_3D[0] + M_3D[1] + 4*M_3D[2])/6 = ",
      (M_3D[0] + M_3D[1] + 4*M_3D[2])/6, "-> tout positif !")
