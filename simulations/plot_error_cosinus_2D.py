from synch_predictions.dynamics.get_reduction_errors import *
# from synch_predictions.plots.plots_setup import *
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

with open('data/cosinus/errors_cosinus/2019_08_20_18h20min40sec_test2_2triangles'
          '_data_parameters_dictionary_for_error_cosinus_2D.json'
          ) as json_data:
    R_dictionary = json.load(json_data)

# with open('data/cosinus/errors_cosinus/2019_08_05_15h52min52sec_50_nodes'
#           '_parameters_for_error_cosinus_2D.json') as json_data:
#     parameters_matrix = np.array(json.load(json_data))
# p_out = parameters_matrix[0]
# omega1 = parameters_matrix[1]
# omega2 = parameters_matrix[2]
# weights = np.ones_like(p_out)/float(len(p_out))
#
# with open('data/cosinus/errors_cosinus/2019_08_05_15h52min52sec_50_nodes'
#           '_dataset_for_error_cosinus_2D.json') as json_data:
#     R_big_matrix = np.array(json.load(json_data))
#
perturbation_list = ["No", "All", "$\\hat{\\mathcal{A}}$",
                     "$\\hat{\\Lambda}$", "Uni"]
keys_array = np.array(["r", "r1", "r2", "r_uni", "r1_uni", "r2_uni",
                       "R_none", "R1_none", "R2_none", "R_all", "R1_all",
                       "R2_all", "R_hatredA", "R1_hatredA", "R2_hatredA",
                       "R_hatLambda", "R1_hatLambda", "R2_hatLambda",
                       "R_uni", "R1_uni", "R2_uni"])
# p_out_array = np.array(R_dictionary["p_out_array"])
omega1_array = np.array(R_dictionary["omega1_array"])
nb_p_out = 1  # len(p_out_array)
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
y_limits_RMSE = None  # [0, 0.05]
y_limits_L1 = None  # [0, 0.05]


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

# ax7 = plt. subplot(4, 3, 7)
# for i in range(5):
#     plt.plot(p_out_array, L1_p_out_matrix[i], linewidth=linewidth,
#              label=perturbation_list[i])
# plt.ylabel("$\\langle L_1 \\rangle_{\\omega_1}$",
#            fontsize=fontsize, labelpad=5)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
# # ylab.set_rotation(0)
# # plt.xlim([0, 1.05])
# plt.ylim(y_limits_L1)
# plt.legend(loc="best", fontsize=fontsize_legend)


# ax8 = plt.subplot(4, 3, 8)
# for i in range(5):
#     plt.plot(p_out_array, L1_1_p_out_matrix[i], linewidth=linewidth,
#              label=perturbation_list[i])
# plt.ylabel("$\\langle (L_1)_1 \\rangle_{\\omega_1}$",
#            fontsize=fontsize, labelpad=5)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
# # ylab.set_rotation(0)
# # plt.xlim([0, 1.05])
# plt.ylim(y_limits_L1)
# plt.legend(loc="best", fontsize=fontsize_legend)

# ax9 = plt.subplot(4, 3, 9)
# for i in range(5):
#     plt.plot(p_out_array, L1_2_p_out_matrix[i], linewidth=linewidth,
#              label=perturbation_list[i])
# plt.ylabel("$\\langle (L_1)_2 \\rangle_{\\omega_1}$",
#            fontsize=fontsize, labelpad=5)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
# # ylab.set_rotation(0)
# # plt.xlim([0, 1.05])
# plt.ylim(y_limits_L1)
# plt.legend(loc="best", fontsize=fontsize_legend)

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

# with open('data/cosinus/errors_cosinus/2019_08_04_03h51min28sec_sim1'
#           '_parameters_for_error_cosinus_2D.json') as json_data:
#     parameters_matrix = np.array(json.load(json_data))
# p_out = parameters_matrix[0]
# omega1 = parameters_matrix[1]
# omega2 = parameters_matrix[2]
# weights = np.ones_like(p_out)/float(len(p_out))
#
# with open('data/cosinus/errors_cosinus/2019_08_04_03h51min28sec_sim1'
#           '_dataset_for_error_cosinus_2D.json') as json_data:
#     R_big_matrix = np.array(json.load(json_data))
#
# perturbation_list = ["No", "All", "$\\hat{\\mathcal{A}}$",
#                      "$\\hat{\\Lambda}$", "Uni"]
# RMSE_list, RMSE1_list, RMSE2_list = get_RMSE_errors(R_big_matrix)
# L1_list, L1_1_list, L1_2_list = get_L1_errors(R_big_matrix)
# L1_matrix, L1_1_matrix, L1_2_matrix =
#  get_L1_errors_vs_parameters(R_big_matrix)
# y_limits_RMSE = [0.14, 0.22]
# y_limits_L1 = [0.05, 0.09]
#
# bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# bins_str = ["$[0.0, 0.2)$", "$[0.2, 0.4)$", "$[0.4, 0.6)$",
#             "$[0.6, 0.8)$", "$[0.8, 1.0)$"]
# nb_bins = len(bins_str)
# nb_points_per_bins = np.zeros(len(bins_str))
# L1_bin_matrix = np.zeros((len(L1_matrix[:, 0]), len(bins_str)))
#
# for j in range(len(p_out)):
#     for i in range(len(bins_str)):
#         if i < len(bins_str)-1:
#             if bins[i] <= p_out[j] < bins[i+1]:
#                 nb_points_per_bins[i] += 1
#                 for k in range(len(L1_matrix[:, 0])):
#                     L1_bin_matrix[k, i] += L1_matrix[k, j]
#
#         else:
#             nb_points_per_bins[i] += 1
#             for k in range(len(L1_matrix[:, 0])):
#                 L1_bin_matrix[k, i] += L1_matrix[k, j]
#
# L1_bin_matrix = L1_bin_matrix/nb_points_per_bins
#
# fig = plt.figure(figsize=(6, 6))
#
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
#
# ax1 = plt.subplot(4, 3, 1)
# plt.bar(perturbation_list, RMSE_list, color="#969696", align='center')
# plt.ylim(y_limits_RMSE)
# plt.ylabel("RMSE", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
#
# ax2 = plt.subplot(4, 3, 2)
# plt.bar(perturbation_list, RMSE1_list, color="#9ecae1", align='center')
# plt.ylim(y_limits_RMSE)
# plt.ylabel("RMSE$_1$", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
# ax3 = plt.subplot(4, 3, 3)
# plt.bar(perturbation_list, RMSE2_list, color="#fdd0a2", align='center')
# plt.ylim(y_limits_RMSE)
# plt.ylabel("RMSE$_2$", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
# ax4 = plt. subplot(4, 3, 4)
# plt.bar(perturbation_list, L1_list, color="#969696", align='center')
# plt.ylim(y_limits_L1)
# plt.ylabel("$\\langle L_1 \\rangle$", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
#
# ax5 = plt.subplot(4, 3, 5)
# plt.bar(perturbation_list, L1_1_list, color="#9ecae1", align='center')
# plt.ylim(y_limits_L1)
# plt.ylabel("$\\langle (L_1)_1 \\rangle$", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
# ax6 = plt.subplot(4, 3, 6)
# plt.bar(perturbation_list, L1_2_list, color="#fdd0a2", align='center')
# plt.ylim(y_limits_L1)
# plt.ylabel("$\\langle (L_1)_2 \\rangle$", fontsize=fontsize, labelpad=5)
# # ylab.set_rotation(0)
#
# ax7 = plt. subplot(4, 3, 7)
# binwidth = 0.0001
# print(len(L1_bin_matrix[i]))
# for i in [4, 3, 2, 1, 0]:
#     plt.bar(bins_str, L1_bin_matrix[i], label=perturbation_list[i]) # , width=0.08)
#     # plt.bar(p_out, L1_matrix[i], label=perturbation_list[i], width=0.08)
# plt.ylabel("$L_1$", fontsize=fontsize, labelpad=5)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
# # ylab.set_rotation(0)
# # plt.xlim([0, 1.05])
# plt.legend(loc="best", fontsize=fontsize_legend)
#
#
# ax8 = plt.subplot(4, 3, 8)
# for i in [4, 3, 2, 1, 0]:
#     # plt.bar(np.abs(omega1-omega2), L1_matrix[i], label=perturbation_list[i],
#     #         width=0.08)
#     plt.scatter(p_out, L1_matrix[i], s=1,
#              label=perturbation_list[i])
# plt.ylabel("$L_1$", fontsize=fontsize, labelpad=5)
# plt.xlabel("$p_{out}$", fontsize=fontsize)   # $|\\omega_1-\\omega_2|$
# # ylab.set_rotation(0)
# # plt.ylim([0, 0.2])
# plt.legend(loc="best", fontsize=fontsize_legend)
#
# ax9 = plt.subplot(4, 3, 9)
# binwidth = 0.01
# for i in [4, 3, 2, 1, 0]:
#     plt.hist(L1_matrix[i], bins=np.arange(min(L1_matrix[i]), max(L1_matrix[i]) + binwidth, binwidth), label=perturbation_list[i])
# plt.ylabel("Count", fontsize=fontsize, labelpad=5)
# plt.xlabel("$L_1$", fontsize=fontsize)
# # ylab.set_rotation(0)
# plt.xlim([0, 1])
# plt.legend(loc="best", fontsize=fontsize_legend)
#
# ax10 = plt.subplot(4, 3, 10)
# plt.hist(p_out, color="#969696", weights=weights)
# plt.ylabel("Count", fontsize=fontsize, labelpad=5)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
# # ylab.set_rotation(0)
# plt.xlim([0, 1.05])
#
#
# ax11 = plt.subplot(4, 3, 11)
# plt.hist(np.abs(omega1-omega2), color="#969696", weights=weights)
# plt.ylabel("Count", fontsize=fontsize, labelpad=5)
# plt.xlabel("$|\\omega_1-\\omega_2|$", fontsize=fontsize)
# # ylab.set_rotation(0)
# plt.ylim([0, 0.2])
#
# ax12 = plt.subplot(4, 3, 12)
# plt.scatter(p_out, np.abs(omega1-omega2), color="#969696", s=1)
# # plt.hist(omega2, color="#969696", weights=weights)
# plt.ylabel("$|\\omega_1-\\omega_2|$", fontsize=fontsize, labelpad=5)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
# # ylab.set_rotation(0)
# plt.xlim([0, 1.05])
#
# plt.tight_layout()
#
# plt.show()
