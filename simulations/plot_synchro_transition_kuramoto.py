from synch_predictions.plots.plot_complete_vs_reduced import *
from synch_predictions.simulations.data_synchro_transition_kuramoto import *
# from synch_predictions.simulations.data import *
import matplotlib.pyplot as plt
import numpy as np
import tkinter.simpledialog
from tkinter import messagebox
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


first_community_color = "#2171b5"
reduced_first_community_color = "#9ecae1"
second_community_color = "#f16913"
reduced_second_community_color = "#fdd0a2"
fontsize = 12
inset_fontsize = 9
fontsize_legend = 12
labelsize = 12
inset_labelsize = 9
linewidth = 2
s = 20
alpha = 0.5
marker = "."
x_lim_kb = [0.08, 0.92]
y_lim_kb = [0, 1.02]
x_lim_ks = [0.08, 0.92]
y_lim_ks = [0, 1.02]
nb_instances = 50

fig = plt.figure(figsize=(6, 3.2))


# Kuramoto on SBM graph
ax = plt.subplot(121)
# ax.title.set_text('$(a)$')
ax.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(np.linspace(0.1, 0.9, 50),
                                     r_kb_matrix, R_kb_matrix,
                                     "#252525", "#969696",
                                     alpha, marker, s, linewidth, nb_instances)
left_bottom_axis_settings(ax, "$p_{out}$", "$\\langle R\,\\rangle_t$",
                          x_lim_kb, y_lim_kb,
                          (0.5, -0.15), (-0.2, 0.45), fontsize,
                          labelsize, linewidth)
plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])

axins1 = inset_axes(ax, width="50%", height="50%",
                    bbox_to_anchor=(.45, .55, .5, .5),   # (-0.12, .7, .5, .5),
                    bbox_transform=ax.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r1_kb_matrix, R1_kb_matrix,
                                     first_community_color,
                                     reduced_first_community_color, alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins1, "$p_{out}$", "$\\langle R_1 \\rangle_t$",
                          x_lim_kb, y_lim_kb, (0.5, -0.2), (-0.25, 0.35),
                          inset_fontsize, inset_labelsize, linewidth)
plt.xticks([0.1, 0.9])


axins2 = inset_axes(ax, width="50%", height="50%",
                    bbox_to_anchor=(.45, .12, .5, .5),  # (.5, .15, .5, .5),
                    bbox_transform=ax.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r2_kb_matrix, R1_kb_matrix,
                                     second_community_color,
                                     reduced_second_community_color, alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins2, "$p_{out}$", "$\\langle R_2 \\rangle_t$",
                          x_lim_kb, y_lim_kb, (0.5, -0.2), (-0.25, 0.35),
                          inset_fontsize, inset_labelsize, linewidth)
plt.xticks([0.1, 0.9])


# Kuramoto on SBM

ax2 = plt.subplot(122)
# ax2.title.set_text('$(b)$')
ax2.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r_ks_matrix, R_ks_matrix,
                                     "#252525", "#969696",
                                     alpha, marker, s, linewidth, nb_instances)
left_bottom_axis_settings(ax2, "$p_{out}$", "$\\langle R \\rangle_t$",
                          x_lim_ks, y_lim_ks,
                          (0.5, -0.15), (-0.2, 0.45), fontsize,
                          labelsize, linewidth)
plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])


axins3 = inset_axes(ax2, width="50%", height="50%",
                    bbox_to_anchor=(-0.08, .15, .5, .5),   # (.5, .55, .5, .5),
                    bbox_transform=ax2.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r1_ks_matrix, R1_ks_matrix,
                                     first_community_color,
                                     reduced_first_community_color, alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins3, "$p_{out}$", "$\\langle R_1 \\rangle_t$",
                          x_lim_ks, y_lim_ks, (0.5, -0.2), (-0.25, 0.35),
                          inset_fontsize, inset_labelsize, linewidth)
plt.xticks([0.1, 0.9])
plt.yticks([0.0, 1.0])


axins4 = inset_axes(ax2, width="50%", height="50%",
                    bbox_to_anchor=(.45, .15, .5, .5),  # (.5, .15, .5, .5),
                    bbox_transform=ax2.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r2_ks_matrix, R2_ks_matrix,
                                     second_community_color,
                                     reduced_second_community_color, alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins4, "$p_{out}$", "$\\langle R_2 \\rangle_t$",
                          x_lim_ks, y_lim_ks, (0.5, -0.2), (-0.25, 0.35),
                          inset_fontsize, inset_labelsize, linewidth)
plt.xticks([0.1, 0.9])
plt.yticks([0.0, 1.0])

plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the plot ?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    fig.savefig("data/kuramoto/"
                "{}_{}_complete_vs_reduced_kuramoto_2D"
                ".png".format(timestr, file))
    fig.savefig("data/kuramoto/"
                "{}_{}_complete_vs_reduced_kuramoto_2D"
                ".pdf".format(timestr, file))

# Old code

# import matplotlib
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# import seaborn as sns
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
# plt.fill_between(p_array, mean_r_Ip1 - std_r_Ip1, mean_r_Ip1 + std_r_Ip1,
#                  color=first_community_color, alpha=0.1)
# plt.plot(p_array, mean_r_Ip1, color=first_community_color,
#          linewidth=linewidth, label=r"Complete")
# plt.errorbar(p_array, mean_R_Ip1, yerr=std_R_Ip1, color="#a8ddb5",
#              linewidth=linewidth - 1, linestyle='--', marker="o",
#              markersize=5, dash_capstyle='butt', dash_joinstyle="bevel",
#              label=r"Reduced")
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
#
# plt.fill_between(p_array, mean_r_I05 - std_r_I05, mean_r_I05 + std_r_I05,
#                  color="#bcbddc", alpha=alpha)
# plt.plot(p_array, mean_r_I05, color="#807dba",
#          linewidth=linewidth, label=r"$I = -0.5$")
# for i in range(0, 10):
#     if i == 0:
#         plt.scatter(p_array, r_kb_matrix[i], color="#252525", s=s, alpha=0.8,
#                     marker=marker)
#     else:
#         plt.scatter(p_array, r_kb_matrix[i], color="#252525", s=s, alpha=0.8,
#                     marker=marker)
#
# plt.fill_between(p_array, mean_R_kb - std_R_kb, mean_R_kb + std_R_kb,
#                  color="#969696", alpha=alpha)
# plt.plot(p_array, mean_R_kb, color="#969696", linewidth=linewidth,
#          linestyle='-')
# sns.set_style("ticks")
# ylab = plt.ylabel("$R$", fontsize=fontsize)
# ylab.set_rotation(0)
# ax.yaxis.set_label_coords(-0.2, 0.45)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
# plt.tick_params(axis='both', which='major', labelsize=labelsize)
# # Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# for axis in ['bottom', 'left']:
#     ax.spines[axis].set_linewidth(1)
# plt.ylim([0, 1.02])
# plt.xlim([0.08, 0.92])
# # plt.legend(bbox_to_anchor=(0.75, 0.95), fontsize=fontsize_legend,
# #            markerscale=2, frameon=False)
# plt.tight_layout()
# for i in range(0, 10):
#     if i == 0:
#         plt.scatter(p_array, r1_kb_matrix[i], color=first_community_color,
#                     s=s, alpha=0.8, marker=marker)
#     else:
#         plt.scatter(p_array, r1_kb_matrix[i], color=first_community_color,
#                     s=s, alpha=0.8, marker=marker)
#
# plt.fill_between(p_array, mean_R1_kb - std_R1_kb, mean_R1_kb + std_R1_kb,
#                  color="#9ecae1", alpha=alpha)
# plt.plot(p_array, mean_R1_kb, color="#9ecae1", linewidth=linewidth,
#          linestyle='-')
# ylab = plt.ylabel("$R_1$", fontsize=inset_fontsize)
# ylab.set_rotation(0)
# plt.xlabel("$p_{out}$", fontsize=inset_fontsize)
# axins1.yaxis.set_label_coords(-0.2, 0.35)
# axins1.xaxis.set_label_coords(0.5, -0.2)
# axins1.spines['right'].set_visible(False)
# axins1.spines['top'].set_visible(False)
# axins1.yaxis.set_ticks_position('left')
# axins1.xaxis.set_ticks_position('bottom')
# plt.tick_params(axis='both', which='major', labelsize=inset_labelsize)
# # axins1.set_xticklabels([])
# for axis in ['bottom', 'left']:
#     axins1.spines[axis].set_linewidth(1)
# plt.ylim([0, 1.02])
# plt.xlim([0.08, 0.92])
# for i in range(0, 10):
#     if i == 0:
#         plt.scatter(p_array, r2_kb_matrix[i], color=second_community_color,
#                     s=s, alpha=0.8, marker=marker)
#     else:
#         plt.scatter(p_array, r2_kb_matrix[i], color=second_community_color,
#                     s=s, alpha=0.8, marker=marker)
#
# plt.fill_between(p_array, mean_R2_kb - std_R1_kb, mean_R2_kb + std_R2_kb,
#                  color="#fdd0a2", alpha=alpha)
# plt.plot(p_array, mean_R2_kb, color="#fdd0a2", linewidth=linewidth,
#          linestyle='-')
# ylab = plt.ylabel("$R_2$", fontsize=inset_fontsize)
# ylab.set_rotation(0)
# axins2.yaxis.set_label_coords(-0.2, 0.35)
# axins2.xaxis.set_label_coords(0.5, -0.2)
# plt.xlabel("$p_{out}$", fontsize=inset_fontsize)
# axins2.spines['right'].set_visible(False)
# axins2.spines['top'].set_visible(False)
# axins2.yaxis.set_ticks_position('left')
# axins2.xaxis.set_ticks_position('bottom')
# for axis in ['bottom', 'left']:
#     axins2.spines[axis].set_linewidth(1)
# plt.tick_params(axis='both', which='major', labelsize=inset_labelsize)
# plt.ylim([0, 1.02])
# plt.xlim([0.08, 0.92])


###############################################################################
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
#                  color="#bcbddc", alpha=alpha)
# plt.plot(p_array, mean_r_I05, color="#807dba",
#          linewidth=linewidth, label=r"$I = -0.5$")
# for i in range(0, 10):
#     if i == 0:
#         plt.scatter(pout_array, r_ks_matrix[i], color="#252525", s=s,
#                     alpha=0.8, marker=marker)
#     else:
#         plt.scatter(pout_array, r_ks_matrix[i], color="#252525", s=s,
#                     alpha=0.8, marker=marker)
#
# plt.fill_between(pout_array, mean_R_ks - std_R_ks, mean_R_ks + std_R_ks,
#                  color="#969696", alpha=alpha)
# plt.plot(pout_array, mean_R_ks, color="#969696", linewidth=linewidth,
#          linestyle='-')
# # sns.set_style("ticks")
# ylab = plt.ylabel("$R$", fontsize=fontsize)
# ylab.set_rotation(0)
# ax2.yaxis.set_label_coords(-0.2, 0.45)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
# plt.tick_params(axis='both', which='major', labelsize=labelsize)
# # Hide the right and top spines
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.yaxis.set_ticks_position('left')
# ax2.xaxis.set_ticks_position('bottom')
# for axis in ['bottom', 'left']:
#     ax2.spines[axis].set_linewidth(1)
# plt.ylim([0.6, 1.02])
# plt.xlim([0, 0.62])
# plt.yticks([0.6, 0.8, 1.0])


# plt.legend(bbox_to_anchor=(0.75, 0.95), fontsize=fontsize_legend,
#            markerscale=2, frameon=False)


# axins3 = inset_axes(ax2, width="50%", height="50%",
#                     bbox_to_anchor=(-0.12, .7, .5, .5),  # (.5, .55, .5, .5),
#                     bbox_transform=ax2.transAxes, loc=4)
# for i in range(0, 10):
#     if i == 0:
#         plt.scatter(pout_array, r1_ks_matrix[i], color=first_community_color,
#                     s=s, alpha=0.8, marker=marker)
#     else:
#         plt.scatter(pout_array, r1_ks_matrix[i], color=first_community_color,
#                     s=s, alpha=0.8, marker=marker)
#
# plt.fill_between(pout_array, mean_R1_ks - std_R1_ks, mean_R1_ks + std_R1_ks,
#                  color="#9ecae1", alpha=alpha)
# plt.plot(pout_array, mean_R1_ks, color="#9ecae1", linewidth=linewidth,
#          linestyle='-')
# ylab = plt.ylabel("$R_1$", fontsize=inset_fontsize)
# ylab.set_rotation(0)
# plt.xlabel("$p_{out}$", fontsize=inset_fontsize)
# plt.tick_params(axis='both', which='major', labelsize=inset_labelsize)
# axins3.yaxis.set_label_coords(-0.2, 0.35)
# axins3.xaxis.set_label_coords(0.5, -0.2)
# axins3.spines['right'].set_visible(False)
# axins3.spines['top'].set_visible(False)
# axins3.yaxis.set_ticks_position('left')
# axins3.xaxis.set_ticks_position('bottom')
# # axins1.set_xticklabels([])
# for axis in ['bottom', 'left']:
#     axins3.spines[axis].set_linewidth(1)
# plt.ylim([0.6, 1.02])
# plt.xlim([0, 0.62])
# axins4 = inset_axes(ax2, width="50%", height="50%",
#                     bbox_to_anchor=(.45, .12, .5, .5),  # (.5, .15, .5, .5),
#                     bbox_transform=ax2.transAxes, loc=4)
# for i in range(0, 10):
#     if i == 0:
#        plt.scatter(pout_array, r2_ks_matrix[i], color=second_community_color,
#                     s=s, alpha=0.8, marker=marker)
#     else:
#        plt.scatter(pout_array, r2_ks_matrix[i], color=second_community_color,
#                     s=s, alpha=0.8, marker=marker)
#
# plt.fill_between(pout_array, mean_R2_ks - std_R2_ks, mean_R2_ks + std_R2_ks,
#                  color="#fdd0a2", alpha=alpha)
# plt.plot(pout_array, mean_R2_ks, color="#fdd0a2", linewidth=linewidth,
#          linestyle='-')
# ylab = plt.ylabel("$R_2$", fontsize=inset_fontsize)
# ylab.set_rotation(0)
# axins4.yaxis.set_label_coords(-0.2, 0.35)
# axins4.xaxis.set_label_coords(0.5, -0.2)
# plt.xlabel("$p_{out}$", fontsize=inset_fontsize)
# axins4.spines['right'].set_visible(False)
# axins4.spines['top'].set_visible(False)
# axins4.yaxis.set_ticks_position('left')
# axins4.xaxis.set_ticks_position('bottom')
# for axis in ['bottom', 'left']:
#     axins4.spines[axis].set_linewidth(1)
# plt.tick_params(axis='both', which='major', labelsize=inset_labelsize)
# plt.ylim([0.6, 1.02])
# plt.xlim([0, 0.62])
# plt.yticks([0.6, 1.0])
# plt.xticks([0.0, 0.6])
