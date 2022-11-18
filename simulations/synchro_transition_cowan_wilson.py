from dynamics.get_synchro_transition import *
from plots.plot_dynamics import *
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
t0, t1, dt = 0, 10, 0.05
time_list = np.linspace(t0, t1, int(t1/dt))

# Structural parameter of the SBM
q = 1
sizes = [50, 50]
N = sum(sizes)
n1 = sizes[0]
n2 = sizes[1]

adjacency_mat = "SBM"

tau = 1
mu = 3

nb_instances = 10
p_array = np.linspace(0.01, 1, 100)

# Get result
mean_x_list, var_x_list, X_list, varX_list = \
    get_synchro_transition_cowan_wilson(p_array, sizes, nb_instances,
                                        t0, t1, dt, tau, mu)


line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "adjacency_matrix_type = Erdos-Renyi\n"
line6 = "nb of graph and initial condition instances = {}" \
        "\n".format(nb_instances)
line7 = "p_array = {}\n".format(p_array)
line8 = "tau = {}\n".format(tau)
line9 = "mu = {}\n".format(mu)
line10 = "x0 = np.random.randn(N)\n"

# with open('synchro_data/theta_model/2019_04_09_13h06min23sec_theta_result'
#           '_complete_r_list.json') as json_data:
#     r_list = np.array(json.load(json_data))
# with open('synchro_data/theta_model/2019_04_09_13h06min23sec_theta_result'
#           '_reduced_R_list.json') as json_data:
#     R_list = np.array(json.load(json_data))

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")


fig = plt.figure(figsize=(10, 8))
# plt.plot(p_array, r_list, color=first_community_color,
#          linestyle='-', linewidth=linewidth, label="$Theoretical$")
# plt.plot(p_array, R_list, color="lightblue",
#          linestyle='-', linewidth=linewidth, label='$Predicted$')
plt.subplot(211)
# plt.scatter(p_array, mean_x_list, s=100, color=first_community_color,
#             label="$Complete$")
# plt.scatter(p_array, X_list, s=50, color="lightblue",
#             label="$Reduced$", marker='s')
plt.plot(p_array, mean_x_list, color=first_community_color,
         linewidth=linewidth, label="$Complete$")
plt.plot(p_array, X_list, color="lightblue", linestyle='--',
         linewidth=linewidth, label="$Reduced$")
plt.ylabel("$X$", fontsize=fontsize)
plt.xlabel("$p$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.ylim([0, 1.02])
plt.xlim([0, 1.02])
plt.legend(loc=2, fontsize=fontsize_legend)
# , ncol=2)# bbox_to_anchor=(1.35, 1.01),

plt.subplot(212)
# plt.scatter(p_array, var_x_list, s=100, color=first_community_color,
#             label="$Complete$")
# plt.scatter(p_array, varX_list, s=50, color="lightblue",
#             label="$Reduced$", marker='s')
plt.plot(p_array, var_x_list, color=first_community_color,
         linewidth=linewidth, label="$Complete$")
plt.plot(p_array, varX_list, color="lightblue", linestyle='--',
         linewidth=linewidth, label="$Reduced$")
plt.ylabel("$\\chi$", fontsize=fontsize)
plt.xlabel("$p$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.ylim([0, 1.02])
plt.xlim([0, 1.02])
plt.legend(loc=2, fontsize=fontsize_legend)
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters, "
                       "the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    fig.savefig("data/cowan_wilson/"
                "{}_{}_complete_vs_reduced_cowan_wilson"
                ".pdf".format(timestr, file))

    f = open('data/cowan_wilson/{}_{}.txt'.format(timestr, file), 'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, "\n", line8,
         line9, line10])

    f.close()

    with open('data/cowan_wilson/{}_{}_complete_mean_x_list'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(mean_x_list, outfile)
    with open('data/cowan_wilson/{}_{}_reduced_X_list'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(X_list, outfile)
    with open('data/cowan_wilson/{}_{}_complete_var_x_list'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(var_x_list, outfile)
    with open('data/cowan_wilson/{}_{}_reduced_varX_list'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(varX_list, outfile)
