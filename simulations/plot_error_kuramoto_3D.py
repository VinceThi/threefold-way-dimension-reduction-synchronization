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

with open('data/kuramoto/errors_kuramoto/2019_09_07_00h21min23sec_omega1_0_2'
          '_omega2_0_6_data_parameters_dictionary_for_error_kuramoto_3D.json'
          ) as json_data:
    R_dictionary = json.load(json_data)

sigma_array = np.array(R_dictionary["sigma_array"])
omega1_array = np.array(R_dictionary["omega1_array"])
nb_sigma = len(sigma_array)
nb_omega1 = len(omega1_array)

r_array = np.array(R_dictionary["r"])
R_array = np.array(R_dictionary["R"])
r_uni_array = np.array(R_dictionary["r_uni"])
R_uni_array = np.array(R_dictionary["R_uni"])

L1_spectral = L1(r_array, R_array)
L1_uniform = L1(r_uni_array, R_uni_array)

y_limits_L1 = [0, 0.5]

# for i in range(len(omega1_array)):
fig = plt.figure(figsize=(8, 8))

ax1 = plt.subplot(2, 2, 1)
plt.plot(sigma_array, r_array[:, 0], color="#252525",
         label="$r_{spec}$")
plt.plot(sigma_array, R_array[:, 0], color="#969696",
         label="$R_{spec}$")
plt.plot(sigma_array, r_uni_array[:, 0], color="#9e9ac8", linestyle="--",
         label="$r_{uni}$")
plt.plot(sigma_array, R_uni_array[:, 0], color="#cbc9e2", linestyle="--",
         label="$R_{uni}$")
plt.ylim([0, 1.05])
plt.ylabel("$R$", fontsize=fontsize, labelpad=5)
plt.xlabel("$\\sigma$", fontsize=fontsize)
plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()

# ax2 = plt.subplot(2, 2, 2)
# plt.plot(sigma_array, r_array[:, 1], color="#252525",
#          label="$R_{spec}$")
# plt.plot(sigma_array, R_array[:, 1], color="#969696",
#          label="$R_{spec}$")
# plt.plot(sigma_array, r_uni_array[:, 1], color="#9e9ac8", linestyle="--",
#          label="$R_{uni}$")
# plt.plot(sigma_array, R_uni_array[:, 1], color="#cbc9e2", linestyle="--",
#          label="$R_{uni}$")
# plt.ylim([0, 1.05])
# # plt.ylabel("$R_{spec}$", fontsize=fontsize, labelpad=5)
# plt.xlabel("$\\sigma$", fontsize=fontsize)
# plt.legend(loc=4, fontsize=fontsize_legend)
# plt.tight_layout()

ax3 = plt.subplot(2, 2, 3)
plt.plot(sigma_array, L1_spectral[:, 0], color="#969696",
         label="Spectral")
plt.plot(sigma_array, L1_uniform[:, 0], color="#cbc9e2",
         label="Uniform")
plt.ylim(y_limits_L1)
plt.ylabel("$L_1$", fontsize=fontsize, labelpad=5)
plt.xlabel("$\\sigma$", fontsize=fontsize)
plt.legend(loc=1, fontsize=fontsize_legend)
plt.tight_layout()

# ax4 = plt.subplot(2, 2, 4)
# plt.plot(sigma_array, L1_spectral[:, 1], color="#969696",
#          label="Spectral")
# plt.plot(sigma_array, L1_uniform[:, 1], color="#cbc9e2",
#          label="Uniform")
# plt.ylim(y_limits_L1)
# plt.ylabel("$L_1$", fontsize=fontsize, labelpad=5)
# plt.xlabel("$\\sigma$", fontsize=fontsize)
# plt.legend(loc=1, fontsize=fontsize_legend)
# plt.tight_layout()

plt.show()
