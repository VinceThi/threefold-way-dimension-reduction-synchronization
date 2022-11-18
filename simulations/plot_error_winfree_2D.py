from dynamics.get_reduction_errors import *
# from plots.plots_setup import *
import matplotlib.pyplot as plt
import json
import numpy as np
first_community_color = "#2171b5"
second_community_color = "#f16913"
fontsize = 12
inset_fontsize = 9
fontsize_legend = 12
labelsize = 12
inset_labelsize = 9
linewidth = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

with open('data/winfree/errors_winfree/2019_08_13_19h45min28sec_small_network'
          '_data_parameters_dictionary_for_error_winfree_2D.json'
          ) as json_data:
    R_dictionary = json.load(json_data)

# with open('data/winfree/errors_winfree/2019_08_05_15h52min52sec_50_nodes'
#           '_parameters_for_error_winfree_2D.json') as json_data:
#     parameters_matrix = np.array(json.load(json_data))
# p_out = parameters_matrix[0]
# omega1 = parameters_matrix[1]
# omega2 = parameters_matrix[2]
# weights = np.ones_like(p_out)/float(len(p_out))
#
# with open('data/winfree/errors_winfree/2019_08_05_15h52min52sec_50_nodes'
#           '_dataset_for_error_winfree_2D.json') as json_data:
#     R_big_matrix = np.array(json.load(json_data))
#
perturbation_list = ["No", "All", "$\\hat{\\mathcal{A}}$",
                     "$\\hat{\\Lambda}$", "Uni"]
keys_array = np.array(["r", "r1", "r2", "r_uni", "r1_uni", "r2_uni",
                       "R_none", "R1_none", "R2_none", "R_all", "R1_all",
                       "R2_all", "R_hatredA", "R1_hatredA", "R2_hatredA",
                       "R_hatLambda", "R1_hatLambda", "R2_hatLambda",
                       "R_uni", "R1_uni", "R2_uni"])
p_out_array = np.array(R_dictionary["p_out_array"])
omega1_array = np.array(R_dictionary["omega1_array"])
nb_p_out = len(p_out_array)
nb_omega1 = len(omega1_array)
R_matrix = get_R_matrix_from_R_dictionary(keys_array, R_dictionary)
RMSE_list, RMSE1_list, RMSE2_list = get_RMSE_errors(R_matrix)
# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)
# print(R_matrix)
L1_list, L1_1_list, L1_2_list = get_mean_L1_errors(R_matrix)
L1_matrix, L1_1_matrix, L1_2_matrix = get_L1_errors(R_matrix)
L1_omega1_matrix, L1_1_omega1_matrix, L1_2_omega1_matrix =\
    get_marginal_L1_errors(0, nb_omega1, nb_p_out,
                           L1_matrix, L1_1_matrix, L1_2_matrix)
L1_p_out_matrix, L1_1_p_out_matrix, L1_2_p_out_matrix =\
    get_marginal_L1_errors(1, nb_omega1, nb_p_out,
                           L1_matrix, L1_1_matrix, L1_2_matrix)
y_limits_RMSE = [0, 0.2]
y_limits_L1 = [0, 0.1]


fig = plt.figure(figsize=(6, 5))

ax1 = plt.subplot(4, 3, 1)
plt.bar(perturbation_list, RMSE_list, color="#969696", align='center')
plt.ylim(y_limits_RMSE)
plt.ylabel("RMSE", fontsize=fontsize, labelpad=5)
# ylab.set_rotation(0)


ax2 = plt.subplot(4, 3, 2)
plt.bar(perturbation_list, RMSE1_list, color="#9ecae1", align='center')
plt.ylim(y_limits_RMSE)
plt.ylabel("RMSE$_1$", fontsize=fontsize, labelpad=5)
# ylab.set_rotation(0)

ax3 = plt.subplot(4, 3, 3)
plt.bar(perturbation_list, RMSE2_list, color="#fdd0a2", align='center')
plt.ylim(y_limits_RMSE)
plt.ylabel("RMSE$_2$", fontsize=fontsize, labelpad=5)
# ylab.set_rotation(0)

ax4 = plt. subplot(4, 3, 4)
plt.bar(perturbation_list, L1_list, color="#969696", align='center')
plt.ylim(y_limits_L1)
plt.ylabel("$\\langle L_1 \\rangle$", fontsize=fontsize, labelpad=5)
# ylab.set_rotation(0)


ax5 = plt.subplot(4, 3, 5)
plt.bar(perturbation_list, L1_1_list, color="#9ecae1", align='center')
plt.ylim(y_limits_L1)
plt.ylabel("$\\langle (L_1)_1 \\rangle$", fontsize=fontsize, labelpad=5)
# ylab.set_rotation(0)

ax6 = plt.subplot(4, 3, 6)
plt.bar(perturbation_list, L1_2_list, color="#fdd0a2", align='center')
plt.ylim(y_limits_L1)
plt.ylabel("$\\langle (L_1)_2 \\rangle$", fontsize=fontsize, labelpad=5)
# ylab.set_rotation(0)

ax7 = plt. subplot(4, 3, 7)
for i in range(5):
    plt.plot(p_out_array, L1_p_out_matrix[i], linewidth=linewidth,
             label=perturbation_list[i])
plt.ylabel("$\\langle L_1 \\rangle_{\\omega_1}$",
           fontsize=fontsize, labelpad=5)
plt.xlabel("$p_{out}$", fontsize=fontsize)
# ylab.set_rotation(0)
# plt.xlim([0, 1.05])
plt.ylim(y_limits_L1)
plt.legend(loc="best", fontsize=fontsize_legend)


ax8 = plt.subplot(4, 3, 8)
for i in range(5):
    plt.plot(p_out_array, L1_1_p_out_matrix[i], linewidth=linewidth,
             label=perturbation_list[i])
plt.ylabel("$\\langle (L_1)_1 \\rangle_{\\omega_1}$",
           fontsize=fontsize, labelpad=5)
plt.xlabel("$p_{out}$", fontsize=fontsize)
# ylab.set_rotation(0)
# plt.xlim([0, 1.05])
plt.ylim(y_limits_L1)
plt.legend(loc="best", fontsize=fontsize_legend)

ax9 = plt.subplot(4, 3, 9)
for i in range(5):
    plt.plot(p_out_array, L1_2_p_out_matrix[i], linewidth=linewidth,
             label=perturbation_list[i])
plt.ylabel("$\\langle (L_1)_2 \\rangle_{\\omega_1}$",
           fontsize=fontsize, labelpad=5)
plt.xlabel("$p_{out}$", fontsize=fontsize)
# ylab.set_rotation(0)
# plt.xlim([0, 1.05])
plt.ylim(y_limits_L1)
plt.legend(loc="best", fontsize=fontsize_legend)

ax10 = plt.subplot(4, 3, 10)
for i in range(5):
    plt.plot(omega1_array, L1_omega1_matrix[i], linewidth=linewidth,
             label=perturbation_list[i])
plt.ylabel("$\\langle L_1 \\rangle_{p_{out}}$",
           fontsize=fontsize, labelpad=5)
plt.xlabel("$\\omega_1$", fontsize=fontsize)
# ylab.set_rotation(0)
# plt.xlim([0, 1.05])
plt.ylim(y_limits_L1)
plt.legend(loc="best", fontsize=fontsize_legend)


ax11 = plt.subplot(4, 3, 11)
for i in range(5):
    plt.plot(omega1_array, L1_1_omega1_matrix[i], linewidth=linewidth,
             label=perturbation_list[i])
plt.ylabel("$\\langle (L_1)_1 \\rangle_{p_{out}}$",
           fontsize=fontsize, labelpad=5)
plt.xlabel("$\\omega_1$", fontsize=fontsize)
# ylab.set_rotation(0)
# plt.xlim([0, 1.05])
plt.ylim(y_limits_L1)
plt.legend(loc="best", fontsize=fontsize_legend)

ax12 = plt.subplot(4, 3, 12)
for i in range(5):
    plt.plot(omega1_array, L1_2_omega1_matrix[i], linewidth=linewidth,
             label=perturbation_list[i])
plt.ylabel("$\\langle (L_1)_2 \\rangle_{p_{out}}$",
           fontsize=fontsize, labelpad=5)
plt.xlabel("$\\omega_1$", fontsize=fontsize)
# ylab.set_rotation(0)
# plt.xlim([0, 1.05])
plt.ylim(y_limits_L1)
plt.legend(loc="best", fontsize=fontsize_legend)

plt.tight_layout()

plt.show()

# ax1 = plt. subplot(331)
# plt.bar(perturbation_list, RMSE_list, color="#969696", align='center')
# plt.ylim(y_limits_RMSE)
# plt.ylabel("RMSE", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
#
# ax2 = plt.subplot(332)
# plt.bar(perturbation_list, RMSE1_list, color="#9ecae1", align='center')
# plt.ylim(y_limits_RMSE)
# plt.ylabel("RMSE$_1$", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
# ax3 = plt.subplot(333)
# plt.bar(perturbation_list, RMSE2_list, color="#fdd0a2", align='center')
# plt.ylim(y_limits_RMSE)
# plt.ylabel("RMSE$_2$", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
# ax4 = plt. subplot(334)
# plt.bar(perturbation_list, L1_list, color="#969696", align='center')
# plt.ylim(y_limits_L1)
# plt.ylabel("$L_1$", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
#
# ax5 = plt.subplot(335)
# plt.bar(perturbation_list, L1_1_list, color="#9ecae1", align='center')
# plt.ylim(y_limits_L1)
# plt.ylabel("$(L_1)_1$", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
# ax6 = plt.subplot(336)
# plt.bar(perturbation_list, L1_2_list, color="#fdd0a2", align='center')
# plt.ylim(y_limits_L1)
# plt.ylabel("$(L_1)_2$", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
# for i in range(2):
#     plt.plot(p_out_array, L1_matrix[i], label=perturbation_list[i],
#              linewidth=linewidth)
# plt.tight_layout()
#
# plt.show()
#
# # print(np.size(r_matrix, axis=1))
# #
# # RMSE_r = RMSE(r_matrix, R_matrix)
# # RMSE_r1 = RMSE(r1_matrix, R1_matrix)
# # RMSE_r2 = RMSE(r2_matrix, R2_matrix)
# #
# # RMSE_r_all = RMSE(r_all_matrix, R_all_matrix)
# # RMSE_r1_all = RMSE(r1_all_matrix, R1_all_matrix)
# # RMSE_r2_all = RMSE(r2_all_matrix, R2_all_matrix)
# #
# # RMSE_r_wA = RMSE(r_wA_matrix, R_wA_matrix)
# # RMSE_r1_wA = RMSE(r1_wA_matrix, R1_wA_matrix)
# # RMSE_r2_wA = RMSE(r2_wA_matrix, R2_wA_matrix)
# #
# # RMSE_r_wLamb = RMSE(r_wLamb_matrix, R_wLamb_matrix)
# # RMSE_r1_wLamb = RMSE(r1_wLamb_matrix, R1_wLamb_matrix)
# # RMSE_r2_wLamb = RMSE(r2_wLamb_matrix, R2_wLamb_matrix)
# #
# # RMSE_p_r = RMSE_vs_p_out(r_matrix, R_matrix)
# # RMSE_p_r1 = RMSE_vs_p_out(r1_matrix, R1_matrix)
# # RMSE_p_r2 = RMSE_vs_p_out(r2_matrix, R2_matrix)
# #
# # RMSE_p_r_all = RMSE_vs_p_out(r_all_matrix, R_all_matrix)
# # RMSE_p_r1_all = RMSE_vs_p_out(r1_all_matrix, R1_all_matrix)
# # RMSE_p_r2_all = RMSE_vs_p_out(r2_all_matrix, R2_all_matrix)
# #
# # RMSE_p_r_wA =  RMSE_vs_p_out(r_wA_matrix, R_wA_matrix)
# # RMSE_p_r1_wA = RMSE_vs_p_out(r1_wA_matrix, R1_wA_matrix)
# # RMSE_p_r2_wA = RMSE_vs_p_out(r2_wA_matrix, R2_wA_matrix)
# #
# # RMSE_p_r_wLamb =  RMSE_vs_p_out(r_wLamb_matrix, R_wLamb_matrix)
# # RMSE_p_r1_wLamb = RMSE_vs_p_out(r1_wLamb_matrix, R1_wLamb_matrix)
# # RMSE_p_r2_wLamb = RMSE_vs_p_out(r2_wLamb_matrix, R2_wLamb_matrix)
# #
# # p_out_array = np.linspace(0.01, 1, 50)
# #
# # fig = plt.figure(figsize=(6, 2))
# #
# # ax1 = plt. subplot(231)
# # plt.bar(["None", "All", "$\\hat{\\mathcal{A}}$", "$\\hat{\\Lambda}$"],
# #         [RMSE_r, RMSE_r_all, RMSE_r_wA, RMSE_r_wLamb], color="#969696",
# #         align='center')
# # #plt.ylim([0.000, 0.008])
# # ylab = plt.ylabel("$d$", fontsize=fontsize, labelpad=15)
# # ylab.set_rotation(0)
# # plt.legend(loc=1, fontsize=fontsize_legend)
# #
# #
# # ax2 = plt.subplot(232)
# # plt.bar(["None", "All", "$\\hat{\\mathcal{A}}$", "$\\hat{\\Lambda}$"],
# #         [RMSE_r1, RMSE_r1_all, RMSE_r1_wA, RMSE_r1_wLamb], color="#9ecae1",
# #         align='center')
# # #plt.ylim([0.000, 0.008])
# # ylab = plt.ylabel("$d_1$", fontsize=fontsize, labelpad=15)
# # ylab.set_rotation(0)
# #
# # ax3 = plt.subplot(233)
# # plt.bar(["None", "All", "$\\hat{\\mathcal{A}}$", "$\\hat{\\Lambda}$"],
# #         [RMSE_r2, RMSE_r2_all, RMSE_r2_wA, RMSE_r2_wLamb], color="#fdd0a2",
# #         align='center')
# # #plt.ylim([0.000, 0.008])
# # ylab = plt.ylabel("$d_2$", fontsize=fontsize, labelpad=15)
# # ylab.set_rotation(0)
# #
# # plt.tight_layout()
# #
# # plt.show()
# #
