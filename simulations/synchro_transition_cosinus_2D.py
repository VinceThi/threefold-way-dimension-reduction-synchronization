from dynamics.get_synchro_transition import *
from plots.plot_dynamics import *
import matplotlib
import numpy as np
import json
import tkinter.simpledialog
from tkinter import messagebox
import time


# Predictions for the cosinus dynamics on modular graphs when one
# module has one natural frequency and the other has another natural frequency


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
first_community_color = "#064878"  # 6, 72, 120
second_community_color = "#ff370f"  # 255, 55, 15
fontsize = 40
fontsize_legend = 30
labelsize = 30
linewidth = 3

# Time parameters
t0, t1, dt = 0, 100, 0.05   # 1000, 0.05
time_list = np.linspace(t0, t1, int(t1 / dt))

# Structural parameter of the SBM
q = 2
sizes = [150, 100]
p_in = 0
mean_SBM = False
N = sum(sizes)
n1 = sizes[0]
n2 = sizes[1]
f = n1/n2

# Dynamical parameters
sigma = 1

omega1 = np.pi        # -np.pi/2
omega2 = -f*omega1   # np.pi/2

omega = np.concatenate([omega1*np.ones(n1), omega2*np.ones(n2)])
# print(np.sum(omega))  # equal to zero if omega2 = -f*omega1
omega_array = np.array([omega1, omega2])

nb_instances = 1  # 50
p_out_array = np.linspace(1/N, 1, 10)

# Get result
r1_matrix, r2_matrix, R1_matrix, R2_matrix, r_matrix, R_matrix = \
    get_synchro_transition_cosinus_2D(p_out_array, p_in, sizes, nb_instances,
                                      t0, t1, dt, omega, omega_array,
                                      sigma, N, mean_SBM=mean_SBM,
                                      plot_temporal_series=1)

line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "Sizes : [n1, n2] = {}\n".format(sizes)
line6 = "adjacency_matrix_type = SBM\n"
line7 = "nb of graph and initial condition instances = {}" \
        "\n".format(nb_instances)
line8 = "p_in = {}\n".format(p_in)
line9 = "p_out_array = {}\n".format(p_out_array)
line10 = "omega1 = {}, omega2 = -n1/n2*omega1= {}\n".format(omega1, omega2)
line11 = "sigma = {}\n".format(sigma)
line12 = "theta0 = 2*np.pi*np.random.randn(N)\n"

# with open('data/cosinus/2019_06_19_00h13min39sec_test2'
#           '_complete_r_matrix_2D.json') as json_data:
#     r_matrix = np.array(json.load(json_data))
# with open('data/cosinus/2019_06_19_00h13min39sec_test2'
#           '_complete_r1_matrix_2D.json') as json_data:
#     r1_matrix = np.array(json.load(json_data))
# with open('data/cosinus/2019_06_19_00h13min39sec_test2'
#           '_complete_r2_matrix_2D.json') as json_data:
#     r2_matrix = np.array(json.load(json_data))
#
#
# with open('data/cosinus/2019_06_19_00h13min39sec_test2'
#           '_reduced_R_matrix_2D.json') as json_data:
#     R_matrix = np.array(json.load(json_data))
# with open('data/cosinus/2019_06_19_00h13min39sec_test2'
#           '_reduced_R1_matrix_2D.json') as json_data:
#     R1_matrix = np.array(json.load(json_data))
# with open('data/cosinus/2019_06_19_00h13min39sec_test2'
#           '_reduced_R2_matrix_2D.json') as json_data:
#     R2_matrix = np.array(json.load(json_data))


mean_r1 = np.mean(r1_matrix, axis=0)
mean_r2 = np.mean(r2_matrix, axis=0)
mean_R1 = np.mean(R1_matrix, axis=0)
mean_R2 = np.mean(R2_matrix, axis=0)
mean_r = np.mean(r_matrix, axis=0)
mean_R = np.mean(R_matrix, axis=0)

std_r1 = np.std(r1_matrix, axis=0)
std_r2 = np.std(r2_matrix, axis=0)
std_R1 = np.std(R1_matrix, axis=0)
std_R2 = np.std(R2_matrix, axis=0)
std_r = np.std(r_matrix, axis=0)
std_R = np.std(R_matrix, axis=0)

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

fig = plt.figure(figsize=(8, 8))
# plt.plot(p_out_array, mean_R1, color="#a8ddb5", linestyle='--',
#          linewidth=linewidth, label="$\\langle R_1\\rangle$")
# plt.plot(p_out_array, mean_R2, color="#feb24c", linestyle='--',
#          linewidth=linewidth, label="$\\langle R_2\\rangle$")

# plt.fill_between(p_out_array, mean_r1-std_r1, mean_r1+std_r1,
#                 color=first_community_color, alpha=0.1)
# plt.fill_between(p_out_array, mean_r2-std_r2, mean_r2+std_r2,
#                 color=second_community_color, alpha=0.1)
# plt.errorbar(p_out_array, mean_r1, yerr=std_r1, color=first_community_color,
# linewidth=1,
#             marker="o", dash_capstyle='butt', dash_joinstyle="bevel",
#             label="$\\langle r_1\\rangle$")
# plt.errorbar(p_out_array, mean_r2, yerr=std_r2, color=second_community_color,
# linewidth=1,
#             marker="o", dash_capstyle='butt', dash_joinstyle="bevel",
#             label="$\\langle r_2\\rangle$")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax0 = plt.subplot(311)
# for i in range(0, len(r_matrix[:, 0])):
#     plt.scatter(p_out_array, r_matrix[i, :], marker="o", alpha=0.2, s=30,
#                 color=first_community_color)
plt.fill_between(p_out_array, mean_r-std_r, mean_r+std_r,
                 color="k", alpha=0.1)
plt.plot(p_out_array, mean_r, color="k",
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_out_array, mean_R, yerr=std_R1, color="#9b9b9b",
             linewidth=linewidth-1, linestyle='--', marker="o", markersize=5,
             dash_capstyle='butt', dash_joinstyle="bevel", label=r"Reduced")
plt.ylabel("$R$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([0, 1.02])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
ax0.set_xticklabels([])

ax1 = plt.subplot(312)
# for i in range(0, len(r1_matrix[:, 0])):
#     plt.scatter(p_out_array, r1_matrix[i, :], marker="o", alpha=0.2, s=30,
#                 color=first_community_color)
plt.fill_between(p_out_array, mean_r1-std_r1, mean_r1+std_r1,
                 color=first_community_color, alpha=0.1)
plt.plot(p_out_array, mean_r1, color=first_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_out_array, mean_R1, yerr=std_R1, color="#a8ddb5",
             linewidth=linewidth-1, linestyle='--', marker="o", markersize=5,
             dash_capstyle='butt', dash_joinstyle="bevel", label=r"Reduced")
plt.ylabel("$R_1$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([0, 1.02])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
ax1.set_xticklabels([])

ax2 = plt.subplot(313)
# for i in range(0, len(r1_matrix[:, 0])):
#     plt.scatter(p_out_array, r2_matrix[i, :], marker="o", alpha=0.2, s=30,
#                 color=second_community_color)
plt.fill_between(p_out_array, mean_r2-std_r2, mean_r2+std_r2,
                 color=second_community_color, alpha=0.1)
plt.plot(p_out_array, mean_r2, color=second_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_out_array, mean_R2, yerr=std_R2, color="#feb24c",
             linewidth=linewidth-1, linestyle='--', marker="o", markersize=5,
             dash_capstyle='butt', dash_joinstyle="bevel", label=r"Reduced")
plt.ylabel("$R_2$", fontsize=fontsize)
plt.xlabel("$p_{out}$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([0, 1.02])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters, "
                       "the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    fig.savefig("data/cosinus/"
                "{}_{}_complete_vs_reduced_cosinus_2D"
                ".png".format(timestr, file))

    f = open('data/cosinus/{}_{}_cosinus_2D.txt'.format(timestr, file), 'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, line8,
         line9, "\n", line10, line11, line12])  # , "\n",
    # "With first order perturbation"])

    f.close()

    with open('data/cosinus/{}_p_in_{}_omega1_{}_n1_{}_n2_{}_{}'
              '_complete_r_matrix_cosinus_2D.json'.format(timestr,
                                                           p_in, omega1,
                                                           n1, n2, file), 'w'
              ) as outfile:
        json.dump(r_matrix.tolist(), outfile)
    with open('data/cosinus/{}_p_in_{}_omega1_{}_n1_{}_n2_{}_{}'
              '_complete_r1_matrix_cosinus_2D.json'.format(timestr,
                                                            p_in, omega1,
                                                            n1, n2, file), 'w'
              ) as outfile:
        json.dump(r1_matrix.tolist(), outfile)
    with open('data/cosinus/{}_p_in_{}_omega1_{}_n1_{}_n2_{}_{}'
              '_complete_r2_matrix_cosinus_2D.json'.format(timestr,
                                                            p_in, omega1,
                                                            n1, n2, file), 'w'
              ) as outfile:
        json.dump(r2_matrix.tolist(), outfile)

    with open('data/cosinus/{}_p_in_{}_omega1_{}_n1_{}_n2_{}_{}'
              '_reduced_R_matrix_cosinus_2D.json'.format(timestr,
                                                          p_in, omega1,
                                                          n1, n2, file), 'w'
              ) as outfile:
        json.dump(R_matrix.tolist(), outfile)
    with open('data/cosinus/{}_p_in_{}_omega1_{}_n1_{}_n2_{}_{}'
              '_reduced_R1_matrix_cosinus_2D.json'.format(timestr,
                                                           p_in, omega1,
                                                           n1, n2, file), 'w'
              ) as outfile:
        json.dump(R1_matrix.tolist(), outfile)
    with open('data/cosinus/{}_p_in_{}_omega1_{}_n1_{}_n2_{}_{}'
              '_reduced_R2_matrix_cosinus_2D.json'.format(timestr,
                                                           p_in, omega1,
                                                           n1, n2, file), 'w'
              ) as outfile:
        json.dump(R2_matrix.tolist(), outfile)
