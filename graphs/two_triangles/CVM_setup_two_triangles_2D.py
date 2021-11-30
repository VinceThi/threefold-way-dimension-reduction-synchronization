from synch_predictions.graphs.get_reduction_matrix_and_characteristics import *
from synch_predictions.graphs.special_graphs import \
    two_triangles_graph_adjacency_matrix

not_aligned = True
A = two_triangles_graph_adjacency_matrix()
graph_str = "two_triangles"
parameters_dictionary = {}
K = np.diag(np.sum(A, axis=1))
# V_K = np.array([[1 / np.sqrt(3), 1 / np.sqrt(3), 0, 0, 1 / np.sqrt(3), 0],
#                 [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]])
V_K = np.array([[1 / np.sqrt(3), 1 / np.sqrt(3), 0, 0, 1 / np.sqrt(3), 0],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]])
# V_K = np.array([[1/2, 1/2, 0, 0, 1/2, 1/2],
#                 [0, 0, 1/np.sqrt(2), 1/np.sqrt(2), 0, 0]])

if not_aligned:
    W = np.diag(np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2]))
    V_W = np.array([[1/np.sqrt(3), 0, 1 / np.sqrt(3), 0, 1 / np.sqrt(3), 0],
                    [0, 1/np.sqrt(3), 0, 1 / np.sqrt(3), 0, 1 / np.sqrt(3)]])
    # W = np.diag(np.array([0.2, -0.4, 0.2, -0.4, 0.2, 0.2]))
    # V_W = np.array([[1/2, 0, 1/2, 0, 1/2, 1/2],
    #                 [0, 1/np.sqrt(2), 0, 1/np.sqrt(2), 0, 0]])
else:
    W = np.diag(np.array([0.2, 0.2, 0.2, -0.2, -0.2, -0.2]))
    V_W = np.array([[1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), 0, 0, 0],
                    [0, 0, 0, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]])

vapvep = np.linalg.eig(A)
V0 = -vapvep[1][:, 0]
# The minus sign is because python gives the dom. eigenvector with negative
# signs for each element in this case.
V1 = vapvep[1][:, 1]
V2 = vapvep[1][:, 2]
V3 = vapvep[1][:, 3]
V4 = vapvep[1][:, 4]
V5 = vapvep[1][:, 5]

V_A = normalize_rows_matrix_VV_T(np.array([V0, V1]))

V_none = np.zeros(np.shape(V_W))

n = V_none.shape[0]
if matrix_is_orthonormalized_VV_T(V_W) \
        and matrix_is_orthonormalized_VV_T(V_K)\
        and matrix_is_orthonormalized_VV_T(V_A):
    get_CVM_dictionary(W, K, A, V_W, V_K, V_A, graph_str,
                       parameters_dictionary,
                       other_procedure=True)
else:
    raise ValueError("One or more eigenvector matrix is not orthonormal.")

""" Old """
# n = len(V_W[:, 0])
#
# print("spec(W) = ", vapW, "\n",
#       "spec(K) = ", vapK, "\n",
#       "spec(A) = ", vapvep[0], "\n \n \n")
#
# CVM_two_triangles_2D_dictionary = {"W": W.tolist(), "K": K.tolist(),
#                                    "A": A.tolist()}
#
# # --------------------------- One target ------------------------------------
#
# """ W """
# T1, T2, T3 = "W", "None", "None"
# V_T1, V_T2, V_T3 = V_W, V_none, V_none
# # C_T1 = np.array([[1 / np.sqrt(3), 0],
# #                 [0, 1 / np.sqrt(3)]])
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
#
# properties_W = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, "None", "None", M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_W"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_W"] = V_T1.tolist()
# CVM_two_triangles_2D_dictionary["M_W"] = M.tolist()
#
#
# """ K """
# T1, T2, T3 = "K", "None", "None"
# V_T1, V_T2, V_T3 = V_K, V_none, V_none
# # C_T1 = np.array([[1 / 2, 0],
# #                 [0, 1 / np.sqrt(2)]])
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_K = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, "None", "None", M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_K"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_K"] = V_T1.tolist()
# CVM_two_triangles_2D_dictionary["M_K"] = M.tolist()
#
#
# """ A """
# T1, T2, T3 = "A", "None", "None"
# V_T1, V_T2, V_T3 = V_A, V_none, V_none
# # C_T1 = np.array([[1, 1],
# #                 [-1, 1]])
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_A = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, "None", "None", M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_A"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_A"] = V_T1.tolist()
# CVM_two_triangles_2D_dictionary["M_A"] = M.tolist()
#
#
# # ---------------------------- Two target -----------------------------------
#
# """ W -> K """
# T1, T2, T3 = "W", "K", "None"
# V_T1, V_T2, V_T3 = V_W, V_K, V_none
# # C_T1 = get_first_target_coefficent_matrix(np.zeros((n, n)), V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_WK = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, "None", M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_WK"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_WK"] = V_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_WK"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["M_WK"] = M.tolist()
#
#
# """ W -> A """
# T1, T2, T3 = "W", "A", "None"
# # In this case, there is a division by zero, because there are elements in
# # VA_VWp_VW that are of opposite sign and then sum to zero.
# V_T1, V_T2, V_T3 = V_W, V_A, V_none
# # C_T1 = get_first_target_coefficent_matrix(np.zeros((n, n)), V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_WA = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, "None", M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_WA"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_WA"] = V_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_WA"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["M_WA"] = M.tolist()
#
#
# """ K -> W """
# T1, T2, T3 = "K", "W", "None"
# V_T1, V_T2, V_T3 = V_K, V_W, V_none
# # C_T1 = get_first_target_coefficent_matrix(np.zeros((n, n)), V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
#
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
#
# properties_KW = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, "None", M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_KW"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_KW"] = V_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_KW"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["M_KW"] = M.tolist()
#
#
# """ K -> A """
# T1, T2, T3 = "K", "A", "None"
# V_T1, V_T2, V_T3 = V_K, V_A, V_none
# # C_T1 = get_first_target_coefficent_matrix(np.zeros((n, n)), V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_KA = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, "None", M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_KA"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_KA"] = V_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_KA"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["M_KA"] = M.tolist()
#
#
# """ A -> W """
# T1, T2, T3 = "A", "W", "None"
# V_T1, V_T2, V_T3 = V_A, V_W, V_none
# # C_T1 = get_first_target_coefficent_matrix(np.zeros((n, n)), V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_AW = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, "None", M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_AW"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_AW"] = V_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_AW"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["M_AW"] = M.tolist()
#
#
# """ A -> K """
# T1, T2, T3 = "A", "K", "None"
# V_T1, V_T2, V_T3 = V_A, V_K, V_none
# # C_T1 = get_first_target_coefficent_matrix(np.zeros((n, n)), V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_AK = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, "None", M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_AK"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_AK"] = V_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_AK"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["M_AK"] = M.tolist()
#
#
# # --------------------------- Three target ----------------------------------
# """ W -> K -> A """
# T1, T2, T3 = "W", "K", "A"
# V_T1, V_T2, V_T3 = V_W, V_K, V_A
# # C_T2 = get_second_target_coefficent_matrix(V_T1, V_T2, V_T3,
# #                                            other_procedure=True)
# # C_T1 = get_first_target_coefficent_matrix(C_T2, V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_WKA = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, V_T3, M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_WKA"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_WKA"] = V_T1.tolist()
# # CVM_two_triangles_2D_dictionary["C_T2_WKA"] = C_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_WKA"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T3_WKA"] = V_T3.tolist()
# CVM_two_triangles_2D_dictionary["M_WKA"] = M.tolist()
#
#
# """ W -> A -> K """
# T1, T2, T3 = "W", "A", "K"
# V_T1, V_T2, V_T3 = V_W, V_A, V_K
# # C_T2 = get_second_target_coefficent_matrix(V_T1, V_T2, V_T3,
# #                                            other_procedure=True)
# # C_T1 = get_first_target_coefficent_matrix(C_T2, V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_WAK =\
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, V_T3, M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_WAK"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_WAK"] = V_T1.tolist()
# # CVM_two_triangles_2D_dictionary["C_T2_WAK"] = C_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_WAK"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T3_WAK"] = V_T3.tolist()
# CVM_two_triangles_2D_dictionary["M_WAK"] = M.tolist()
#
#
# """ K -> W -> A """
# T1, T2, T3 = "K", "W", "A"
# V_T1, V_T2, V_T3 = V_K, V_W, V_A
# # C_T2 = get_second_target_coefficent_matrix(V_T1, V_T2, V_T3,
# #                                            other_procedure=True)
# # C_T1 = get_first_target_coefficent_matrix(C_T2, V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_KWA = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, V_T3, M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_KWA"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_KWA"] = V_T1.tolist()
# # CVM_two_triangles_2D_dictionary["C_T2_KWA"] = C_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_KWA"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T3_KWA"] = V_T3.tolist()
# CVM_two_triangles_2D_dictionary["M_KWA"] = M.tolist()
#
#
# """ K -> A -> W """
# T1, T2, T3 = "K", "A", "W"
# V_T1, V_T2, V_T3 = V_K, V_A, V_W
# # C_T2 = get_second_target_coefficent_matrix(V_T1, V_T2, V_T3,
# #                                            other_procedure=True)
# # C_T1 = get_first_target_coefficent_matrix(C_T2, V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_KAW =\
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, V_T3, M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_KAW"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_KAW"] = V_T1.tolist()
# # CVM_two_triangles_2D_dictionary["C_T2_KAW"] = C_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_KAW"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T3_KAW"] = V_T3.tolist()
# CVM_two_triangles_2D_dictionary["M_KAW"] = M.tolist()
#
#
# """ A -> W -> K """
# T1, T2, T3 = "A", "W", "K"
# V_T1, V_T2, V_T3 = V_A, V_W, V_K
# # C_T2 = get_second_target_coefficent_matrix(V_T1, V_T2, V_T3,
# #                                            other_procedure=True)
# # C_T1 = get_first_target_coefficent_matrix(C_T2, V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_AWK = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, V_T3, M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_AWK"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_AWK"] = V_T1.tolist()
# # CVM_two_triangles_2D_dictionary["C_T2_AWK"] = C_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_AWK"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T3_AWK"] = V_T3.tolist()
# CVM_two_triangles_2D_dictionary["M_AWK"] = M.tolist()
#
#
# """ A -> K -> W """
# T1, T2, T3 = "A", "K", "W"
# V_T1, V_T2, V_T3 = V_A, V_K, V_W
# # C_T2 = get_second_target_coefficent_matrix(V_T1, V_T2, V_T3,
# #                                            other_procedure=True)
# # C_T1 = get_first_target_coefficent_matrix(C_T2, V_T1, V_T2)
# # M = get_reduction_matrix(C_T1, V_T1)
# M = get_reduction_matrix(V_T1, V_T2, V_T3)
# properties_AKW = \
#     get_properties_and_errors_for_different_targets(T1, T2, T3,
#                                                     V_T1, V_T2, V_T3, M,
#                                                     W, K, A)
# # CVM_two_triangles_2D_dictionary["C_T1_AKW"] = C_T1.tolist()
# CVM_two_triangles_2D_dictionary["V_T1_AKW"] = V_T1.tolist()
# # CVM_two_triangles_2D_dictionary["C_T2_AKW"] = C_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T2_AKW"] = V_T2.tolist()
# CVM_two_triangles_2D_dictionary["V_T3_AKW"] = V_T3.tolist()
# CVM_two_triangles_2D_dictionary["M_AKW"] = M.tolist()
#
# properties_list = \
#     ["One target", properties_W, properties_K, properties_A,
#      "Two target", properties_WK, properties_WA, properties_KW,
#      properties_KA, properties_AW, properties_AK,
#      "Three target", properties_WKA, properties_WAK, properties_KWA,
#      properties_KAW, properties_AWK, properties_AKW]
#
# if messagebox.askyesno("Python",
#                        "Would you like to save the dictionary: "
#                        "CVM_two_triangles_2D_dictionary?"):
#     window = tkinter.Tk()
#     window.withdraw()  # hides the window
#     file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
#
#     timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
#
#     f = open('CVM_data/{}_CVM_two_triangles_2D_properties_{}'
#              '.txt'.format(timestr, file), 'w')
#     f.writelines(properties_list)
#
#     f.close()
#
#     with open('CVM_data/{}_CVM_two_triangles_2D_dictionary_{}'
#               '.json'.format(timestr, file), 'w') as outfile:
#         json.dump(CVM_two_triangles_2D_dictionary, outfile)
