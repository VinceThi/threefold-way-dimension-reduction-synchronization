from dynamics.get_reduction_errors import *
# from plots.plots_setup import *
import matplotlib.pyplot as plt
import json
import numpy as np

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
labelpad = 10
x, y = 0.02, 0.02
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

with open('data/kuramoto/errors_kuramoto/2019_09_19_12h03min42sec_500_points_'
          'data_parameters_dictionary_for_error_kuramoto_small_bipartite_2D'
          '.json'
          ) as json_data:
    R_dictionary = json.load(json_data)

deg_freq = 1
if deg_freq:
    # Degree-freq corrrelation data on small SBM  3D
    # V0, V1, V5 as eigenvec
    # with open('data/kuramoto/errors_kuramoto/2019_09_20_11h58min52sec_'
    #           'deg_freq_500'
    #           '_points_data_parameters_dictionary_for_error_kuramoto_small_'
    #           'bipartite_3D.json'
    #           ) as json_data:
    #     R_dictionary_3D = json.load(json_data)

    # Degree-freq corrrelation data on small SBM  3D
    # V0, V1, V2 as eigenvec
    with open('data/kuramoto/errors_kuramoto/2020_01_14_23h39min30sec_degree'
              '_freq_corr_data_parameters_dictionary_for_error_kuramoto_'
              'small_bipartite_3D.json'
              ) as json_data:
        R_dictionary_3D = json.load(json_data)

else:
    # without divergence, V0, V1, V5 as eigenvec (fig (j))
    # Small SBM 3D
    # with open('data/kuramoto/errors_kuramoto/2019_09_20_09h53min39sec_500_'
    #           'points_averaging5_data_parameters_dictionary_'
    #           'for_error_kuramoto_small_bipartite_3D.json'
    #           ) as json_data:
    #     R_dictionary_3D = json.load(json_data)

    # with divergence, V0, V1, V2 as eigenvec(fig (j))
    # Small SBM 3D
    with open('data/kuramoto/errors_kuramoto/2020_01_14_22h06min33sec_'
              'with_divergence_data_parameters_dictionary_for_error_'
              'kuramoto_small_bipartite_3D.json'
              ) as json_data:
        R_dictionary_3D = json.load(json_data)


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
RMSE_spectral = RMSE(r_array, R_array)
RMSE_uniform = RMSE(r_uni_array, R_uni_array)
# print(RMSE_spectral, RMSE_uniform)

sigma_array_3D = np.array(R_dictionary_3D["sigma_array"])
omega1_array_3D = np.array(R_dictionary_3D["omega1_array"])
nb_sigma_3D = len(sigma_array_3D)
nb_omega1_3D = len(omega1_array_3D)

r_array_3D = np.array(R_dictionary_3D["r"])
R_array_3D = np.array(R_dictionary_3D["R"])
r_uni_array_3D = np.array(R_dictionary_3D["r_uni"])
R_uni_array_3D = np.array(R_dictionary_3D["R_uni"])

L1_spectral_3D = L1(r_array_3D, R_array_3D)
L1_uniform_3D = L1(r_uni_array_3D, R_uni_array_3D)
RMSE_spectral_3D = RMSE(r_array_3D, R_array_3D)
RMSE_uniform_3D = RMSE(r_uni_array_3D, R_uni_array_3D)
# print(RMSE_spectral_3D, RMSE_uniform_3D)

# Find the intersection where spectral reduction 3D becomes better than
# uniform freq reduction 3D
# diff_array = L1_uniform[:, 0]-L1_spectral_3D[:, 0]
# j = -1
# for i in range(len(L1_uniform)):
#     if diff_array[j-i] < 0.0005:
#         print(j-i)
#         break
# print(sigma_array[j-i])
# for i in range(len(omega1_array)):

fig = plt.figure(figsize=(6, 6))

ax1 = plt.subplot(2, 2, 1)
plt.plot(sigma_array, r_uni_array[:, 0],
         color=total_color, linestyle="-",
         label="$r_{freq}$")
plt.plot(sigma_array, R_uni_array[:, 0],
         color=reduced_first_community_color, linestyle="-",
         label="$R_{freq}$")
plt.fill_between(sigma_array, r_uni_array[:, 0], R_uni_array[:, 0],
                 color=total_color, alpha=0.2)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.text(x=x, y=y, s="$M = M_W$ \n RMSE $\\approx$ {}"
         .format("%.3f" % np.round(RMSE_uniform, 3)), fontsize=fontsize)
plt.ylim([0, 1.02])
plt.xlim([-0.05, 3.05])
plt.xticks([0, 1, 2, 3])
plt.xlabel("$\\sigma$", fontsize=fontsize)
plt.legend(loc=4, fontsize=fontsize_legend, handlelength=0.9)
plt.tight_layout()

ax2 = plt.subplot(2, 2, 2)
plt.plot(sigma_array_3D, r_uni_array_3D[:, 0], color=total_color,
         label="$r_{deg}$")
plt.plot(sigma_array_3D, R_uni_array_3D[:, 0],
         color=reduced_second_community_color, linestyle="-",
         label="$R_{deg}$")
plt.fill_between(sigma_array_3D, r_uni_array_3D[:, 0], R_uni_array_3D[:, 0],
                 color=total_color, alpha=0.2)
if deg_freq:
    plt.text(x=x, y=y, s="$M = M_K = M_W$ \n RMSE $\\approx$ {}"
             .format("%.3f" % np.round(RMSE_uniform_3D, 3)), fontsize=fontsize)
else:
    plt.text(x=x, y=y, s="$M = M_K$ \n RMSE $\\approx$ {}"
             .format("%.3f" % np.round(RMSE_uniform_3D, 3)), fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.vlines(3.963, ymin=0, ymax=0.84205, linestyle="--", color="#bdbdbd")
plt.ylim([0, 1.02])
plt.xlim([-0.05, 3.05])
plt.xticks([0, 1, 2, 3])
plt.xlabel("$\\sigma$", fontsize=fontsize)
plt.legend(loc=4, fontsize=fontsize_legend, handlelength=0.9)
plt.tight_layout()


ax3 = plt.subplot(2, 2, 3)
plt.plot(sigma_array, r_array[:, 0], color=total_color,
         label="$r_{spec}$")
plt.plot(sigma_array, R_array[:, 0],
         color=reduced_third_community_color, linestyle="-",
         label="$R_{spec}$")
plt.fill_between(sigma_array, r_array[:, 0], R_array[:, 0],
                 color=total_color, alpha=0.2)
plt.text(x=x, y=y, s="$M = M_A$ \n $M_T = M_W$ \n"
                     "RMSE $\\approx$ {}"
         .format("%.3f" % np.round(RMSE_spectral, 3)), fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([0, 1.02])
plt.xlim([-0.05, 3.05])
plt.xticks([0, 1, 2, 3])
plt.xlabel("$\\sigma$", fontsize=fontsize)
plt.legend(loc=4, fontsize=fontsize_legend, handlelength=0.9)
plt.tight_layout()

ax4 = plt.subplot(2, 2, 4)
plt.plot(sigma_array_3D, r_array_3D[:, 0], color=total_color,
         label="$r_{spec}$")
plt.plot(sigma_array_3D, R_array_3D[:, 0],
         color=reduced_third_community_color, linestyle="-",
         label="$R_{spec}$")
plt.fill_between(sigma_array_3D, r_array_3D[:, 0], R_array_3D[:, 0],
                 color=total_color, alpha=0.2)
if deg_freq:
    plt.text(x=x, y=y, s="$M = M_A$ \n $M_T = M_K = M_W$ \n"
                         "RMSE $\\approx$ {}"
             .format("%.3f" % np.round(RMSE_spectral_3D, 3)),
             fontsize=fontsize)
else:
    plt.text(x=x, y=y, s="$M = M_A$ \n $M_T = M_K$ \n"
                         "RMSE $\\approx$ {}"
             .format("%.3f" % np.round(RMSE_spectral_3D, 3)),
             fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([0, 1.02])
plt.xlabel("$\\sigma$", fontsize=fontsize)
plt.xlim([-0.05, 3.05])
plt.xticks([0, 1, 2, 3])
plt.legend(loc=4, fontsize=fontsize_legend, handlelength=0.9)
plt.tight_layout()

# ax3 = plt.subplot(2, 2, 4)
# plt.scatter([1, 2, 3], [0, 0.012345679012345684, 0.012345679012345678],
#             marker="o", s=100, color=reduced_first_community_color)
# plt.scatter([1, 2, 3], [0.0011111111111111113, 0, 0.05555555555555555],
#             marker="+", s=100, color=reduced_second_community_color)
# plt.scatter([1, 2, 3], [0.0011111111111111111, 0.0036512737712722804, 0],
#             marker="x", s=100, color=reduced_third_community_color)
# plt.plot(sigma_array, L1_uniform[:, 0],
#          color=reduced_first_community_color, label="$(L_1)_{freq}$")
# plt.plot(sigma_array_3D, L1_uniform_3D[:, 0],
#          color=reduced_second_community_color, label="$(L_1)_{deg}$")
# plt.plot(sigma_array_3D, L1_spectral_3D[:, 0],
#          color=reduced_third_community_color, label="$(L_1)_{spec}$")
#
# plt.ylim([0, 0.5])
# ylab = plt.ylabel("$L_1$", fontsize=fontsize, labelpad=labelpad)
# ylab.set_rotation(0)
# ax3.set_xticklabels(["", "$W$", "", "$K$", "", "$A$"])
# plt.xlabel("$\\sigma$", fontsize=fontsize)
# plt.legend(loc=1, fontsize=fontsize_legend, handlelength=0.9)
# plt.tight_layout()

plt.show()

# For statistical physics ... random code
# fig = plt.figure(figsize=(3, 3))
# x = np.linspace(0, 1, 1000)
# y = x
# x1 = 0.5
# x2 = 0.9
# ax1 = plt.subplot(2, 1, 1)
# plt.plot(x, 3*x**2, color=reduced_first_community_color,
#          linewidth=linewidth)
# plt.vlines(x1, ymin=0, ymax=3*x1**2, linestyle="--", color="#bdbdbd")
# plt.vlines(x2, ymin=0, ymax=3*x2**2, linestyle="--", color="#bdbdbd")
# plt.ylabel("$\\rho_X(x)$", fontsize=fontsize)
# plt.xlabel("$x \\in R_X$", fontsize=fontsize)
# plt.ylim([0, 3.1])
#
# ax2 = plt.subplot(2, 1, 2)
# plt.plot(y, 3/2*y**(1/2), color=reduced_second_community_color,
#          linewidth=linewidth)
# plt.vlines(x1**2, ymin=0, ymax=3/2*x1, linestyle="--", color="#bdbdbd")
# plt.vlines(x2**2, ymin=0, ymax=3/2*x2, linestyle="--", color="#bdbdbd")
# plt.ylabel("$\\rho_Y(y)$", fontsize=fontsize)
# plt.xlabel("$y \\in R_Y$", fontsize=fontsize)
# plt.ylim([0, 3.1])
# plt.tight_layout()
# plt.show()
