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

# TODO
# TODO
# TODO !
# TODO: La somme n'est pas la bonne ici pour la dynamique complète!
# On devrait sommer sur les module B_mu à la place !  Voir la fonction
# pour cowan-wilson dans get_synchro_transition.

# Time parameters
t0, t1, dt = 0, 10, 0.05
time_list = np.linspace(t0, t1, int(t1/dt))

# Structural parameter of the SBM
q = 2
sizes = [150, 100]
N = sum(sizes)
n1 = sizes[0]
n2 = sizes[1]


tau = 1
mu = 4

nb_instances = 10
p_out_array = np.linspace(0.01, 1, 50)
p_11 = 0
p_22 = 0

# Get result
mean_x1_list, mean_x2_list, r1_list, r2_list, \
    X1_list, X2_list, R1_list, R2_list = \
    get_synchro_transition_cowan_wilson_2D(p_out_array, p_11, p_22, sizes,
                                           nb_instances, t0, t1, dt,
                                           tau, mu)

line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "adjacency_matrix_type = SBM\n"
line6 = "nb of graph and initial condition instances = {}" \
        "\n".format(nb_instances)
line7 = "p_array = {}\n".format(p_out_array)
line8 = "p_11 = {}\n".format(p_11)
line9 = "p_22 = {}\n".format(p_22)
line10 = "tau = {}\n".format(tau)
line11 = "mu = {}\n".format(mu)
line12 = "x0 = np.random.randn(N)\n"

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
plt.plot(p_out_array, mean_x1_list, color=first_community_color,
         linewidth=linewidth, label="$Complete\:1$")
plt.plot(p_out_array, mean_x2_list, color=second_community_color,
         linewidth=linewidth, label="$Complete\:2$")
plt.plot(p_out_array, X1_list, color="#a8ddb5", linestyle='--',
         linewidth=linewidth, label="$Reduced\:1$")
plt.plot(p_out_array, X2_list, color="#feb24c", linestyle='--',
         linewidth=linewidth, label="$Reduced\:2$")
plt.ylabel("$X$", fontsize=fontsize)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.ylim([0, 1.02])
plt.xlim([0, 1.02])
plt.legend(loc=2, fontsize=fontsize_legend-4)
# , ncol=2)# bbox_to_anchor=(1.35, 1.01),

plt.subplot(212)
# plt.scatter(p_array, var_x_list, s=100, color=first_community_color,
#             label="$Complete$")
# plt.scatter(p_array, varX_list, s=50, color="lightblue",
#             label="$Reduced$", marker='s')
# plt.plot(p_out_array, var_x1_list, color=first_community_color,
#          linewidth=linewidth, label="$Complete\:1$")
# plt.plot(p_out_array, var_x2_list, color=second_community_color,
#          linewidth=linewidth, label="$Complete\:2$")
# plt.plot(p_out_array, varX1_list, color="#a8ddb5", linestyle='--',
#          linewidth=linewidth, label="$Reduced\:1$")
# plt.plot(p_out_array, varX2_list, color="#feb24c", linestyle='--',
#          linewidth=linewidth, label="$Reduced\:2$")
plt.plot(p_out_array, r1_list, color=first_community_color,
         linewidth=linewidth, label="$Complete\:1$")
plt.plot(p_out_array, r2_list, color=second_community_color,
         linewidth=linewidth, label="$Complete\:2$")
plt.plot(p_out_array, R1_list, color="#a8ddb5", linestyle='--',
         linewidth=linewidth, label="$Reduced\:1$")
plt.plot(p_out_array, R2_list, color="#feb24c", linestyle='--',
         linewidth=linewidth, label="$Reduced\:2$")
plt.ylabel("$R$", fontsize=fontsize)
plt.xlabel("$p_{out}$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([0, 1.02])
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
                ".png".format(timestr, file))

    f = open('data/cowan_wilson/{}_{}.txt'.format(timestr, file), 'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, line8,
         line9, "\n", line10, line11, line12])

    f.close()

    with open('data/cowan_wilson/{}_{}_complete_mean_x1_list'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(mean_x1_list, outfile)
    with open('data/cowan_wilson/{}_{}_complete_mean_x2_list'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(mean_x2_list, outfile)
    with open('data/cowan_wilson/{}_{}_reduced_X1_list'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(X1_list, outfile)
    with open('data/cowan_wilson/{}_{}_reduced_X2_list'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(X2_list, outfile)
    # with open('data/cowan_wilson/{}_{}_complete_var_x1_list'
    #           '.json'.format(timestr, file), 'w') as outfile:
    #     json.dump(var_x1_list, outfile)
    # with open('data/cowan_wilson/{}_{}_complete_var_x2_list'
    #           '.json'.format(timestr, file), 'w') as outfile:
    #     json.dump(var_x2_list, outfile)
    # with open('data/cowan_wilson/{}_{}_reduced_varX1_list'
    #           '.json'.format(timestr, file), 'w') as outfile:
    #     json.dump(varX1_list, outfile)
    # with open('data/cowan_wilson/{}_{}_reduced_varX2_list'
    #           '.json'.format(timestr, file), 'w') as outfile:
    #     json.dump(varX2_list, outfile)
    with open('data/cowan_wilson/{}_{}_complete_r1_list'       
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(r1_list, outfile)
    with open('data/cowan_wilson/{}_{}_complete_r2_list'       
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(r2_list, outfile)
    with open('data/cowan_wilson/{}_{}_reduced_R1_list'         
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(R1_list, outfile)
    with open('data/cowan_wilson/{}_{}_reduced_R2_list'         
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(R2_list, outfile)
