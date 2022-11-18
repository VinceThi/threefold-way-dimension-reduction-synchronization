from dynamics.get_reduction_errors import *
# from plots.plots_setup import *
import matplotlib.pyplot as plt
import json
import numpy as np
# from tqdm import tqdm
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
s = 50
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


with open('data/kuramoto/errors_kuramoto/2019_10_02_17h44min36sec_big_sim_data'
          '_parameters_dictionary_for_error_kuramoto_2D_vs_N.json'
          ) as json_data:
    R_big_dictionary = json.load(json_data)

p_out_array_big = np.array(R_big_dictionary["p_out_array"])
omega1_array_big = np.array(R_big_dictionary["omega1_array"])
N_array_big = np.array(R_big_dictionary["N_array"])
nb_p_out_big = len(p_out_array_big)
nb_omega1_big = len(omega1_array_big)
plot_transitions_big = 0

L1_spec_vs_N_big, var_L1_spec_vs_N_big = \
    get_error_vs_N_kuramoto_bipartite(R_big_dictionary, p_out_array_big,
                                      omega1_array_big, N_array_big,
                                      plot_transitions_big)

with open('data/kuramoto/errors_kuramoto/2019_10_03_15h29min13sec_'
          'test_data_parameters_dictionary_for_error_kuramoto_2D_vs_N.json'
          ) as json_data:
    R_dictionary = json.load(json_data)

p_out_array = np.array(R_dictionary["p_out_array"])
omega1_array = np.array(R_dictionary["omega1_array"])
N_array = np.array(R_dictionary["N_array"])
nb_p_out = len(p_out_array)
nb_omega1 = len(omega1_array)
plot_transitions = 0

L1_spec_vs_N, var_L1_spec_vs_N = \
    get_error_vs_N_kuramoto_bipartite(R_dictionary, p_out_array, omega1_array,
                                      N_array, plot_transitions)

with open('data/kuramoto/errors_kuramoto/2019_10_11_14h25min01sec_N5000_data_'
          'parameters_dictionary_for_error_kuramoto_2D_vs_N.json'
          ) as json_data:
    R_dictionary_5000 = json.load(json_data)

p_out_array_5000 = np.array(R_dictionary_5000["p_out_array"])
omega1_array_5000 = np.array(R_dictionary_5000["omega1_array"])
N_array_5000 = np.array(R_dictionary_5000["N_array"])
plot_transitions_5000 = 0

L1_spec_vs_N5000, var_L1_spec_vs_N5000 = \
    get_error_vs_N_kuramoto_bipartite(R_dictionary_5000, p_out_array_5000,
                                      omega1_array_5000,
                                      N_array_5000, plot_transitions_5000)

with open('data/kuramoto/errors_kuramoto/2019_10_14_07h38min20sec_N10000_data_'
          'parameters_dictionary_for_error_kuramoto_2D_vs_N.json'
          ) as json_data:
    R_dictionary_10000 = json.load(json_data)

p_out_array_10000 = np.array(R_dictionary_10000["p_out_array"])
omega1_array_10000 = np.array(R_dictionary_10000["omega1_array"])
N_array_10000 = np.array(R_dictionary_10000["N_array"])
plot_transitions_10000 = 0

L1_spec_vs_N10000, var_L1_spec_vs_N10000 = \
    get_error_vs_N_kuramoto_bipartite(R_dictionary_10000, p_out_array_10000,
                                      omega1_array_10000,
                                      N_array_10000, plot_transitions_10000)

plt.figure(figsize=(6, 3))
ax1 = plt.subplot(121)
# plt.scatter(N_array, RMSE_freq_vs_N, label="RMSE$_{freq}$", s=s,
#             color=reduced_first_community_color)
# plt.scatter(N_array, RMSE_spec_vs_N, label="RMSE$_{spec}$", s=s,
#             color=reduced_third_community_color)
# plt.errorbar(N_array, L1_freq_vs_N, yerr=var_L1_freq_vs_N, fmt='o',
#              color=reduced_first_community_color,
#              label="$\\langle L_1 \\rangle_{freq}$")
plt.title("(a)", fontsize=fontsize)
plt.errorbar(N_array_big, L1_spec_vs_N_big, yerr=var_L1_spec_vs_N_big, fmt='o',
             color=reduced_third_community_color,
             label="$\\langle L_1 \\rangle_{spec}$")
plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.legend(loc=1, fontsize=fontsize_legend)
ylab = plt.ylabel("$\\langle L_1 \\rangle$", fontsize=fontsize, labelpad=15)
ylab.set_rotation(0)
plt.xlabel("$N$", fontsize=fontsize)
plt.xscale('symlog')
plt.ylim([0, 0.1])

ax2 = plt.subplot(122)
# plt.scatter(N_array, RMSE_freq_vs_N, label="RMSE$_{freq}$", s=s,
#             color=reduced_first_community_color)
# plt.scatter(N_array, RMSE_spec_vs_N, label="RMSE$_{spec}$", s=s,
#             color=reduced_third_community_color)
# plt.errorbar(N_array, L1_freq_vs_N, yerr=var_L1_freq_vs_N, fmt='o',
#              color=reduced_first_community_color,
#              label="$\\langle L_1 \\rangle_{freq}$")
plt.title("(b)", fontsize=fontsize)
plt.errorbar(N_array, L1_spec_vs_N, yerr=var_L1_spec_vs_N, fmt='o',
             color=reduced_third_community_color,
             label="$\\langle L_1 \\rangle_{spec}$")
plt.errorbar(N_array_5000, L1_spec_vs_N5000, yerr=var_L1_spec_vs_N5000,
             fmt='o', color=reduced_third_community_color)
plt.errorbar(N_array_10000, L1_spec_vs_N10000, yerr=var_L1_spec_vs_N10000,
             fmt='o', color=reduced_third_community_color)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.legend(loc=1, fontsize=fontsize_legend)
ylab = plt.ylabel("$\\langle L_1 \\rangle$", fontsize=fontsize, labelpad=15)
ylab.set_rotation(0)
plt.xlabel("$N$", fontsize=fontsize)
plt.xscale('symlog')
plt.ylim([0, 0.4])


# axins1 = inset_axes(ax, width="70%", height="70%",
#                   bbox_to_anchor=(.45, .50, .5, .5),   # (-0.12, .7, .5, .5),
#                   bbox_transform=ax.transAxes, loc=4)
# # plt.errorbar(N_array, L1_freq_vs_N, yerr=var_L1_freq_vs_N, fmt='o',
# #              color=reduced_first_community_color)
# plt.errorbar(N_array, L1_spec_vs_N, yerr=var_L1_spec_vs_N, fmt='o',
#              color=reduced_third_community_color)
# plt.xlim([600, 10550])
# plt.ylim([0, 0.005])
# plt.xticks([1000, 5000, 10000])
# plt.tick_params(axis='both', which='major', labelsize=labelsize)
# ylab = plt.ylabel("$\\langle L_1 \\rangle$", fontsize=fontsize, labelpad=15)
# ylab.set_rotation(0)
# plt.xlabel("$N$", fontsize=fontsize)

plt.tight_layout()

plt.show()
