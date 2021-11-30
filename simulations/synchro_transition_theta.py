from synch_predictions.dynamics.get_synchro_transition import *
from synch_predictions.plots.plot_dynamics import *
import matplotlib
import numpy as np
import json
import tkinter.simpledialog
from tkinter import messagebox
import time


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
first_community_color = "#064878"   # 6, 72, 120
second_community_color = "#ff370f"  # 255, 55, 15
fontsize = 40
fontsize_legend = 30
labelsize = 30
linewidth = 3


# Time parameters
t0 = 0
t1 = 100  
dt = 0.05 

# Structural parameter of the SBM (Erdos-Renyi)
q = 1
sizes = [300, 200]   # N = 500, the asymmetry doesn't matter for ER
N = sum(sizes)
n1 = sizes[0]
n2 = sizes[1]

adjacency_mat = "SBM"

sigma = 4
Iext = -1

nb_instances = 50
p_array = np.linspace(0.01, 1, 50)

# Get result
r_matrix, R_matrix = get_synchro_transition_theta(p_array, sizes, nb_instances,
                                                  t0, t1, dt, Iext, sigma, N,
                                                  plot_temporal_series=0)

line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "adjacency_matrix_type = Erdos-Renyi\n"
line6 = "nb of graph and initial condition instances = {}" \
        "\n".format(nb_instances)
line7 = "p_array = {}\n".format(p_array)
line8 = "sigma = {}\n".format(sigma)
line9 = "Iext = {}\n".format(Iext)
line10 = "theta0 = 2*np.pi*np.random.randn(N)\n"


# with open('data/theta/2019_07_03_13h31min17sec_sigma_4_I_minus0_25'
#           '_complete_r_matrix.json') as json_data:
#     r_matrix = np.array(json.load(json_data))
# with open('data/theta/2019_07_03_13h31min17sec_sigma_4_I_minus0_25'
#           '_reduced_R_matrix.json') as json_data:
#     R_matrix = np.array(json.load(json_data))
# p_array = np.linspace(0.001, 1, 50)

mean_r = np.mean(r_matrix, axis=0)
mean_R = np.mean(R_matrix, axis=0)
std_r = np.std(r_matrix, axis=0)
std_R = np.std(R_matrix, axis=0)

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

fig = plt.figure(figsize=(8, 5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#  plt.plot(p_array, r_list, color=first_community_color,
#           linestyle='-', linewidth=linewidth, label="$Theoretical$")
#  plt.plot(p_array, R_list, color="lightblue",
#           linestyle='-', linewidth=linewidth, label='$Predicted$')
# plt.scatter(p_array, mean_r, s=100, color=first_community_color,
#             label="$Complete$")
# plt.scatter(p_array, mean_R, s=50, color="lightblue",
#             label="$Reduced$", marker='s')
# plt.ylabel("$R$", fontsize=fontsize)
# plt.xlabel("$p$", fontsize=fontsize)
# plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.ylim([0, 1.02])
# plt.xlim([0, 1.02])
# plt.legend(loc=3, fontsize=fontsize_legend)
# # , ncol=2)# bbox_to_anchor=(1.35, 1.01),
# plt.show()
plt.fill_between(p_array, mean_r-std_r, mean_r+std_r,
                 color=first_community_color, alpha=0.1)
plt.plot(p_array, mean_r, color=first_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_array, mean_R, yerr=std_R, color="#a8ddb5",
             linewidth=linewidth-1, linestyle='--', marker="o", markersize=5,
             dash_capstyle='butt', dash_joinstyle="bevel", label=r"Reduced")
plt.ylabel("$R$", fontsize=fontsize)
plt.xlabel("$p$", fontsize=fontsize)
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

    fig.savefig("data/theta/"
                "{}_{}_complete_vs_reduced_theta_model"
                ".png".format(timestr, file))

    f = open('data/theta/{}_{}_theta_1D.txt'.format(timestr, file), 'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, "\n", line8,
         line9, line10])

    f.close()

    with open('data/theta/{}_{}_complete_r_matrix'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(r_matrix.tolist(), outfile)
    with open('data/theta/{}_{}_reduced_R_matrix'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(R_matrix.tolist(), outfile)
