from synch_predictions.graphs.get_reduction_matrix_and_characteristics import *
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

# ---------------------- One target W, n = 2 ----------------------------------
T1, T2, T3 = "W", "None", "None"
C_W = np.array([[1/np.sqrt(3), 0],
                [0, 1/np.sqrt(3)]])
V_W = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
                [0, 0, 0, 1/np.sqrt(3),  1/np.sqrt(3), 1/np.sqrt(3)]])
M_W = np.array([[1/3, 1/3, 1/3, 0, 0, 0],
                [0, 0, 0, 1/3,  1/3, 1/3]])
# print(M_W, C_W@V_W)  # Verification
get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_W, "None", "None",
                                                C_W, np.array([[0]]), M_W,
                                                W, K, A)

# ----------------------- One target W, n = 3 ---------------------------------
T1, T2, T3 = "W", "None", "None"
C_W = np.array([[1/np.sqrt(3), 0, 0],
                [0, 1/np.sqrt(2), 0],
                [0, 0, 1]])
V_W = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
                [0, 0, 0, 1/np.sqrt(2),  1/np.sqrt(2), 0],
                [0, 0, 0, 0, 0, 1]])
M_W = np.array([[1/3, 1/3, 1/3, 0, 0, 0],
                [0, 0, 0, 1/2, 1/2, 0],
                [0, 0, 0, 0, 0, 1]])
# print(M_W, C_W@V_W)  # Verification
get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_W, "None", "None",
                                                C_W, np.array([[0]]), M_W,
                                                W, K, A)

# ----------------------------- One target K ----------------------------------
T1, T2, T3 = "K", "None", "None"
C_K = np.array([[1/np.sqrt(2), 0, 0],
                [0, 1/np.sqrt(2), 0],
                [0, 0, 1/np.sqrt(2)]])
V_K = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0, 0],
                [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
                [0, 0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]])
M_K = np.array([[1/2, 1/2, 0, 0, 0, 0],
                [0, 0, 1/2, 1/2, 0, 0],
                [0, 0, 0, 0, 1/2, 1/2]])
# print(M_K, C_K@V_K)  # Verification
get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_K, "None", "None",
                                                C_K, np.array([[0]]), M_K,
                                                W, K, A)

# --------------------------- Two target A then W -----------------------------
T1, T2, T3 = "A", "W", "None"
V0 = -vapvep[1][:, 0]
# The minus sign is because python gives the dom. eigenvector with negative
# signs for each element.
V1 = vapvep[1][:, 1]
V2 = vapvep[1][:, 2]
V3 = vapvep[1][:, 3]
V4 = vapvep[1][:, 4]
V5 = vapvep[1][:, 5]
V_A = np.array([V0, V1])
# print("V^+ V = ", np.sum(np.dot(pinv(V_2D), V_2D), axis=1), "\n\n\n")

C_A = M_W@pinv(V_A)
C_A_V_A = C_A@V_A
M_AW = (C_A_V_A.T / (np.sum(C_A_V_A, axis=1))).T

get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_A, V_W, "None",
                                                C_A, np.array([[0]]), M_AW,
                                                W, K, A)

# ----------------------- Two target A then K -----------------------------
T1, T2, T3 = "A", "K", "None"
V_A = np.array([V0, V1, V3])
C_A = M_K@pinv(V_A)

C_A_V_A = C_A@V_A

M_AK = (C_A_V_A.T / (np.sum(C_A_V_A, axis=1))).T

# print(np.linalg.svd(M_AK))
# plt.matshow(M_AK, aspect="auto")
# plt.colorbar()
# plt.show()
#
# plt.matshow(np.linalg.svd(M_AK)[-1], aspect="auto")
# plt.colorbar()
# plt.show()
#
# print(np.linalg.det(C_A))

get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_A, V_K, "None",
                                                C_A, np.array([[0]]), M_AK,
                                                W, K, A)

# ----------------------- Two target W then A -----------------------------
T1, T2, T3 = "W", "A", "None"
V_W = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
                [0, 0, 0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])
V_A = np.array([V0, V1])
C_W = V_A@pinv(V_W)
CW_VW = C_W@V_W
M_WA = CW_VW  # .T # / (np.sum(CW_VW, axis=1))).T
# In this case, there is a division by zero, because there are elements in
# VA_VWp_VW that are of opposite sign and then sum to zero.

get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_W, V_A, "None",
                                                C_W, np.array([[0]]), M_WA,
                                                W, K, A)

# ---------------------- Two target K then W ------------------------------
T1, T2, T3 = "K", "W", "None"
V_W_3D = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])

V_K_3D = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0, 0],
                   [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
                   [0, 0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]])
C_K = V_W_3D @ pinv(V_K_3D)
CK_VK = C_K @ V_K_3D
M_KW = (CK_VK.T / (np.sum(CK_VK, axis=1))).T

get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_K_3D, V_W_3D, "None",
                                                C_K, np.array([[0]]), M_KW,
                                                W, K, A)

# ---------------------- Two target K then W ------------------------------
T1, T2, T3 = "K", "W", "None"
V_W_3D = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0],
                   [0, 0, 0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])

V_K_3D = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0, 0],
                   [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
                   [0, 0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]])

C_K = V_W_3D @ pinv(V_K_3D)
CK_VK = C_K @ V_K_3D
M_KW = (CK_VK.T / (np.sum(CK_VK, axis=1))).T
print("Other VEPs")
get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_K_3D, V_W_3D, "None",
                                                C_K, np.array([[0]]), M_KW,
                                                W, K, A)

# ---------------------- Two target K then W ------------------------------
T1, T2, T3 = "K", "W", "None"
V_W_3D = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
                   [0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
                   [0, 0, 0, 0, 0, 1]])

V_K_3D = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0, 0],
                   [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
                   [0, 0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]])

C_K = V_W_3D @ pinv(V_K_3D)
CK_VK = C_K @ V_K_3D
M_KW = (CK_VK.T / (np.sum(CK_VK, axis=1))).T
print("Other VEPs 2")
get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_K_3D, V_W_3D, "None",
                                                C_K, np.array([[0]]), M_KW,
                                                W, K, A)

# --------------- Three target K then W then A ----------------------------
T1, T2, T3 = "K", "W", "A"
V_W_3D = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
                   [0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
                   [0, 0, 0, 0, 0, 1]])

V_K_3D = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0, 0],
                   [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
                   [0, 0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]])

V_A = np.array([V0, V1, V3])

C_W = V_A@pinv(V_W_3D)
C_K = C_W@V_W_3D @ pinv(V_K_3D)
CK_VK = C_K @ V_K_3D
M_KWA = CK_VK  # .T / (np.sum(CK_VK, axis=1))).T

get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_K_3D, V_W_3D, V_A,
                                                C_K, C_W, M_KWA,
                                                W, K, A)

# -------------- Three target W then A then K,  n = 3 ---------------------
T1, T2, T3 = "W", "A", "K"
V_W_3D = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
                   [0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
                   [0, 0, 0, 0, 0, 1]])

V_K_3D = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0, 0],
                   [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
                   [0, 0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]])

V_A = np.array([V0, V1, V3])

# print(np.linalg.svd(CW_VW))
# print(V_A@pinv(V_W_3D)@V_W_3D@pinv(V_A@pinv(V_W_3D)@V_W_3D))

C_A = V_K_3D @ pinv(V_A)
# C_A = V_K_3D @ pinv(V_W_3D) @ V_W_3D @ pinv(V_A) # other procedure
C_W = C_A @ V_A @ pinv(V_W_3D)
CW_VW = C_W @ V_W_3D
M_WAK = (CW_VW.T / (np.sum(CW_VW, axis=1))).T

print("n = 3")
get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_W_3D, V_A, V_K_3D,
                                                C_W, C_A, M_WAK,
                                                W, K, A)

# -------------- Three target W then A then K,  n = 2 ---------------------
T1, T2, T3 = "W", "A", "K"
V_W_2D = np.array(
    [[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
     [0, 0, 0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])

V_K_2D = np.array([[1/2, 1/2, 0, 0, 1/2, 1/2],
                   [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0]])

V_A = np.array([V0, V1])

# print(V_A@pinv(V_W_2D)@V_W_2D@pinv(V_A@pinv(V_W_2D)@V_W_2D))

C_A = V_K_2D @ pinv(V_A)
# C_A = V_K_2D @ pinv(V_W_2D) @ V_W_2D @ pinv(V_A) # other procedure
C_W = C_A @ V_A @ pinv(V_W_2D)
CW_VW = C_W @ V_W_2D
M_WAK = (CW_VW.T / (np.sum(CW_VW, axis=1))).T
print("n = 2: not good result, M_1j = M_2j for all j !")
get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_W_2D, V_A, V_K_2D,
                                                C_W, C_A, M_WAK,
                                                W, K, A)

# -------------- Three target A then K then W,  n = 3 ---------------------
T1, T2, T3 = "A", "K", "W"
V_W_3D = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
                   [0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
                   [0, 0, 0, 0, 0, 1]])

V_K_3D = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0, 0, 0],
                   [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
                   [0, 0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]])

V_A = np.array([V0, V1, V3])

C_K = V_W_3D @ pinv(V_K_3D)
# C_K = V_W_3D @ pinv(V_A) @ V_A @ pinv(V_K_3D)  # other procedure(good here)
C_A = C_K @ V_K_3D @ pinv(V_A)
CA_VA = C_A @ V_A
M_AKW = (CA_VA.T / (np.sum(CA_VA, axis=1))).T

get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_A, V_K_3D, V_W_3D,
                                                C_A, C_K, M_AKW,
                                                W, K, A)

# -------------- Three target W then K then A,  n = 3 ---------------------
T1, T2, T3 = "W", "K", "A"
V_W_3D = np.array(
    [[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
     [0, 0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
     [0, 0, 0, 0, 0, 1]])

V_K_3D = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0],
                   [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
                   [0, 0, 0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)]])

V_A = np.array([V0, V1, V3])

# C_K = V_A @ pinv(V_K_3D)
C_K = V_A @ pinv(V_W_3D) @ V_W_3D @ pinv(V_K_3D)  # other procedure
C_W = C_K @ V_K_3D @ pinv(V_W_3D)
CW_VW = C_W @ V_W_3D
M_WKA = (CW_VW.T / (np.sum(CW_VW, axis=1))).T

get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_W_3D, V_K_3D, V_A,
                                                C_W, C_K, M_WKA,
                                                W, K, A)

# -------------- Three target W then K then A,  n = 2 ---------------------
T1, T2, T3 = "W", "K", "A"
V_W_2D = np.array(
    [[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
     [0, 0, 0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])

V_K_2D = np.array([[1/2, 1/2, 0, 0, 1/2, 1/2],
                   [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0]])

V_A = np.array([V0, V1])

# C_K = V_A @ pinv(V_K_2D)
C_K = V_A @ pinv(V_W_2D) @ V_W_2D @ pinv(V_K_2D)  # other procedure
C_W = C_K @ V_K_2D @ pinv(V_W_2D)
CW_VW = C_W @ V_W_2D
M_WKA = (CW_VW.T / (np.sum(CW_VW, axis=1))).T

get_properties_and_errors_for_different_targets(T1, T2, T3,
                                                V_W_2D, V_K_2D, V_A,
                                                C_W, C_K, M_WKA,
                                                W, K, A)
