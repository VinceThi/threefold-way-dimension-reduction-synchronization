# from synch_predictions.plots.plot_dynamics import *
from simulations.data_synchro_transition_theta import *
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tkinter.simpledialog
from tkinter import messagebox
import time
# import seaborn as sns


# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
first_community_color = "#2171b5"
second_community_color = "#f16913"
fontsize = 12
fontsize_legend = 12
labelsize = 12
linewidth = 2
s = 30
alpha = 0.5

p_array = np.linspace(0.001, 1, 50)

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

fig = plt.figure(figsize=(5, 4))
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# plt.fill_between(p_array, mean_r_Ip1 - std_r_Ip1, mean_r_Ip1 + std_r_Ip1,
#                  color=first_community_color, alpha=0.1)
# plt.plot(p_array, mean_r_Ip1, color=first_community_color,
#          linewidth=linewidth, label=r"Complete")
# plt.errorbar(p_array, mean_R_Ip1, yerr=std_R_Ip1, color="#a8ddb5",
#              linewidth=linewidth - 1, linestyle='--', marker="o",
#              markersize=5, dash_capstyle='butt', dash_joinstyle="bevel",
#              label=r"Reduced")

ax = plt.subplot(111)

# for i in range(0, 10):
#     if i == 0:
#         plt.scatter(p_array[1:], r_Ip1_matrix[i][1:], color="#bcbddc", s=s,
#                     label=r"$I = 1$")
#     else:
#         plt.scatter(p_array[1:], r_Ip1_matrix[i][1:], color="#bcbddc", s=s)
#
# plt.fill_between(p_array[1:], mean_R_Ip1[1:] - std_R_Ip1[1:],
#                 mean_R_Ip1[1:] + std_R_Ip1[1:], color="#9e9ac8", alpha=alpha)
# plt.plot(p_array, mean_R_Ip1, color="#9e9ac8", linewidth=linewidth,
#          linestyle='-')

# plt.fill_between(p_array, mean_r_I05 - std_r_I05, mean_r_I05 + std_r_I05,
#                 color="#bcbddc", alpha=alpha)
# plt.plot(p_array, mean_r_I05, color="#807dba",
#         linewidth=linewidth, label=r"$I = -0.5$")
for i in range(0, 10):
    if i == 0:
        plt.scatter(p_array, r_I05_matrix[i], color="#efedf5", s=s, alpha=0.8,
                    label=r"$I = -0.5$", marker="2")
    else:
        plt.scatter(p_array, r_I05_matrix[i], color="#efedf5", s=s, alpha=0.8,
                    marker="2")

plt.fill_between(p_array, mean_R_I05 - std_R_I05, mean_R_I05 + std_R_I05,
                 color="#807dba", alpha=alpha)
plt.plot(p_array, mean_R_I05, color="#807dba", linewidth=linewidth,
         linestyle='-')
# plt.errorbar(p_array, mean_R_I05, yerr=std_R_I05, color="#6baed6",
#            linewidth=linewidth - 1, linestyle='--', marker="o", markersize=5,
#            dash_capstyle='butt', dash_joinstyle="bevel", label=r"Reduced")

# plt.fill_between(p_array, mean_r_I1 - std_r_I1, mean_r_I1 + std_r_I1,
#                  color=first_community_color, alpha=0.1)
# plt.plot(p_array, mean_r_I1, color=first_community_color,
#          linewidth=linewidth, label=r"$I = -1$")
for i in range(0, 10):
    if i == 0:
        plt.scatter(p_array, r_I1_matrix[i],  color="#c6dbef", s=s,
                    label=r"$I = -1$", marker="1")
    else:
        plt.scatter(p_array, r_I1_matrix[i], color="#c6dbef", s=s,
                    marker="1")
plt.fill_between(p_array, mean_R_I1 - std_R_I1, mean_R_I1 + std_R_I1,
                 color="#6baed6", alpha=alpha)
plt.plot(p_array, mean_r_I1, color="#6baed6", linewidth=linewidth,
         linestyle='-')
# plt.errorbar(p_array, mean_R_I1, yerr=std_R_I1, color="#9ecae1",
#            linewidth=linewidth - 1, linestyle='--', marker="o", markersize=5,
#              dash_capstyle='butt', dash_joinstyle="bevel", label=r"Reduced")

# plt.fill_between(p_array, mean_r_I2 - std_r_I2, mean_R_I2 + std_r_I2,
#                  color="#4292c6", alpha=0.1)
# plt.plot(p_array, mean_r_I2, color="#4292c6",
#          linewidth=linewidth, label=r"$I = -2$")
for i in range(0, 10):
    if i == 0:
        plt.scatter(p_array, r_I2_matrix[i], color="#e0f3db", s=s,
                    label=r"$I = -2$", marker="+")
    else:
        plt.scatter(p_array, r_I2_matrix[i], color="#e0f3db", s=s,
                    marker="+")

plt.fill_between(p_array, mean_R_I2 - std_R_I2, mean_R_I2 + std_R_I2,
                 color="#7bccc4", alpha=alpha)
plt.plot(p_array, mean_R_I2, color="#7bccc4",
         linewidth=linewidth, linestyle='-')
# plt.errorbar(p_array, mean_R_I2, yerr=std_R_I2, color="#c6dbef",
#            linewidth=linewidth - 1, linestyle='--', marker="o", markersize=5,
#            dash_capstyle='butt', dash_joinstyle="bevel", label=r"$I = -2$")

# sns.set_style("ticks")
ylab = plt.ylabel("$R$", fontsize=fontsize)
ylab.set_rotation(0)
ax.yaxis.set_label_coords(-0.15, 0.45)
plt.xlabel("$p$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1)
plt.ylim([0, 1.05])
plt.xlim([0, 1.02])
plt.legend(bbox_to_anchor=(0.75, 0.95), fontsize=fontsize_legend,
           markerscale=2, frameon=False)
plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the plot ?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    fig.savefig("data/theta/"
                "{}_{}_complete_vs_reduced_theta_model"
                ".png".format(timestr, file))
    fig.savefig("data/theta/"
                "{}_{}_complete_vs_reduced_theta_model"
                ".pdf".format(timestr, file))
