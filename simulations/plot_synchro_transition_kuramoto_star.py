from plots.plot_complete_vs_reduced import *
from simulations.data_synchro_transition_kuramoto import *
import matplotlib.pyplot as plt
import numpy as np
# import tkinter.simpledialog
# from tkinter import messagebox
# import time
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes


first_community_color = "#2171b5"
second_community_color = "#f16913"
reduced_first_community_color = "#9ecae1"
reduced_second_community_color = "#fdd0a2"
reduced_third_community_color = "#a1d99b"
reduced_fourth_community_color = "#9e9ac8"
total_color = "#525252"
fontsize = 12
inset_fontsize = 9
fontsize_legend = 12
labelsize = 12
inset_labelsize = 9
linewidth = 2
s = 20
alpha = 0.5
marker = "."
x_lim_kb = [0.78, 2.52]
y_lim_kb = [0, 1.02]

# with open('data/kuramoto/2019_10_03_17h05min34sec_small_test_data
#           _R_dictionary_'
#           'kuramoto_star_2D.json'
#           ) as json_data:
#     R_dictionary = json.load(json_data)
with open('data/kuramoto/2019_10_09_11h26min38sec_hysteresis_data_R_dictionary'
          '_kuramoto_star_2D.json'
          ) as json_data:
    R_dictionary = json.load(json_data)


sigma_array = np.array(R_dictionary["sigma_array"])
omega_array = np.array(R_dictionary["omega_array"])

nb_sigma = len(sigma_array)
nb_init_cond = 2

r_s_matrix = np.array(R_dictionary["r"]).T
R_s_matrix = np.array(R_dictionary["R"]).T


def R_glob(Rp, Phi, N):
    return np.absolute(np.exp(1j*Phi) + (N-1)*Rp)/N


def Phi_fixed_point_1(omega1, omega2, sigma, N):
    return np.arcsin((omega1 - omega2) / (sigma * N))


Rp_fixed_point_1 = 1
sigma_array_fp1 = np.linspace(0.5, 2.5, 10000)


def R_top_branch(omega1, omega2, sigma, N):
    return np.sqrt(1 + (N-1)**2 +
                   2*(N-1)*np.sqrt(1 - ((omega1 - omega2)/(sigma*N))**2))/N


def R_branch(omega1, omega2, sigma, N):
    return np.sqrt(1 + (N-1)**2 -
                   2*(N-1)*np.sqrt(1 - ((omega1 - omega2)/(sigma*N))**2))/N


def Rp_fixed_point_2(omega1, omega2, sigma, N):
    return np.sqrt((2*(omega1 - omega2) - sigma) / (sigma * (2*N - 1)))


Phi_fixed_point_2 = np.pi/2
sigma_array_fp2 = np.linspace(0.81, 2.5, 10000)


def R_middle_branch(omega1, omega2, sigma, N):
    return np.sqrt(1 + (N-1)**2*(
            (2*(omega1 - omega2) - sigma)/(sigma*(2*N - 1))))/N


def R_unknown_branch(omega1, omega2, sigma, N):
    return (np.sqrt((2*(omega1 - omega2) - sigma)/(sigma*(2*N - 1))) - 1)/N


def R_unknown_branch_2(omega1, omega2, sigma, N):
    return (np.sqrt((2*(omega1 - omega2) - sigma)/(sigma*(2*N - 1))) + 1)/N


fig = plt.figure(figsize=(4, 4))

# Kuramoto on star graph
ax = plt.subplot(111)
# ax.title.set_text('$(a)$')
# ax.title.set_position((0.5, 1.05))
colors_complete = ["#252525", reduced_third_community_color]
colors_reduced = ["#969696", reduced_fourth_community_color]
for i in range(nb_init_cond):
    plt.scatter(sigma_array, r_s_matrix[i], color=colors_complete[i],
                marker=marker, s=s+80, linewidth=linewidth)

    plt.scatter(sigma_array, R_s_matrix[i], color=colors_reduced[i],
                marker=marker, s=s, linewidth=linewidth)

plt.plot(sigma_array_fp1, R_top_branch(10, 1, sigma_array_fp1, 11),
         linewidth=linewidth-1, color="r", linestyle="-")
plt.plot(sigma_array_fp1, R_branch(10, 1, sigma_array_fp1, 11),
         linewidth=linewidth-1, color="b", linestyle="--")
# plt.plot(sigma_array_fp2, R_middle_branch(10, 1, sigma_array_fp2, 11),
#          linewidth=linewidth-1, color="g", linestyle="-")
# plt.plot(sigma_array_fp2, R_unknown_branch(10, 1, sigma_array_fp1, 11),
#          linewidth=linewidth-1, color="#252525", linestyle="-")
# plt.plot(sigma_array_fp2, R_unknown_branch_2(10, 1, sigma_array_fp1, 11),
#          linewidth=linewidth-1, color="#252525", linestyle="-")

# plot_transitions_complete_vs_reduced(sigma_array,
#                                      r_s_matrix, R_s_matrix,
#                                      "#252525", "#969696",
#                                      alpha, marker, s, linewidth,
#                                      nb_instances)
left_bottom_axis_settings(ax, "$\\sigma$", "$R$", x_lim_kb, y_lim_kb,
                          (0.5, -0.15), (-0.2, 0.45), fontsize,
                          labelsize, linewidth)
plt.xticks([0.5, 1, 1.5, 2, 2.5])

plt.tight_layout()
plt.show()
# if messagebox.askyesno("Python",
#                        "Would you like to save the plot ?"):
#     window = tkinter.Tk()
#     window.withdraw()  # hides the window
#     file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
#     timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
#     # fig.savefig("data/kuramoto/"
#     #             "{}_{}_complete_vs_reduced_kuramoto_star_2D"
#     #             ".png".format(timestr, file))
#     fig.savefig("data/kuramoto/"
#                 "{}_{}_complete_vs_reduced_kuramoto_star_2D"
#                 ".pdf".format(timestr, file))
