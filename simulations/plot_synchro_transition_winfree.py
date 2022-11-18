from plots.plot_complete_vs_reduced import *
from simulations.data_synchro_transition_winfree import *
import matplotlib.pyplot as plt
import numpy as np
import tkinter.simpledialog
from tkinter import messagebox
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


first_community_color = "#2171b5"
second_community_color = "#f16913"
fontsize = 12
inset_fontsize = 9
fontsize_legend = 12
labelsize = 12
inset_labelsize = 9
linewidth = 2
s = 20
alpha = 0.5
marker = "."
x_lim_wb = [0, 1.02]
y_lim_wb = [0, 1.02]
x_lim_ws = [0, 1.02]
y_lim_ws = [0, 1.02]
nb_instances = 50

fig = plt.figure(figsize=(6, 3.2))

# Winfree on SBM graph
ax = plt.subplot(121)
# ax.title.set_text('$(a)$')
ax.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r_wb_matrix, R_wb_matrix,
                                     "#252525", "#969696",
                                     alpha, marker, s, linewidth, nb_instances)
left_bottom_axis_settings(ax, "$p_{out}$", "$\\langle R\\rangle_t$", x_lim_wb, y_lim_wb,
                          (0.5, -0.15), (-0.2, 0.45), fontsize,
                          labelsize, linewidth)

axins1 = inset_axes(ax, width="50%", height="50%",
                    bbox_to_anchor=(-0.08, .7, .5, .5),   # (.5, .55, .5, .5),
                    bbox_transform=ax.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r1_wb_matrix, R1_wb_matrix,
                                     first_community_color,
                                     "#9ecae1", alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins1, "$p_{out}$", "$\\langle R_1 \\rangle_t$",
                          x_lim_wb, y_lim_wb, (0.5, -0.2), (-0.25, 0.35),
                          inset_fontsize, inset_labelsize, linewidth)
plt.xticks([0, 1])


axins2 = inset_axes(ax, width="50%", height="50%",
                    bbox_to_anchor=(.5, .15, .5, .5), 
                    bbox_transform=ax.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r2_wb_matrix, R1_wb_matrix,
                                     second_community_color,
                                     "#fdd0a2", alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins2, "$p_{out}$", "$\\langle R_2 \\rangle_t$",
                          x_lim_wb, y_lim_wb, (0.5, -0.2), (-0.25, 0.35),
                          inset_fontsize, inset_labelsize, linewidth)
plt.xticks([0, 1])


# Winfree on SBM

ax2 = plt.subplot(122)
# ax2.title.set_text('$(b)$')
ax2.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r_ws_matrix, R_ws_matrix,
                                     "#252525", "#969696",
                                     alpha, marker, s, linewidth, nb_instances)
left_bottom_axis_settings(ax2, "$p_{out}$", "$\\langle R \\rangle_t$", x_lim_ws, y_lim_ws,
                          (0.5, -0.15), (-0.2, 0.45), fontsize,
                          labelsize, linewidth)


axins3 = inset_axes(ax2, width="50%", height="50%",
                    bbox_to_anchor=(-0.08, .15, .5, .5),
                    bbox_transform=ax2.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r1_ws_matrix, R1_ws_matrix,
                                     first_community_color,
                                     "#9ecae1", alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins3, "$p_{out}$", "$\\langle R_1 \\rangle_t$",
                          x_lim_ws, y_lim_ws, (0.5, -0.2), (-0.25, 0.35),
                          inset_fontsize, inset_labelsize, linewidth)
plt.xticks([0.0, 1.0])
plt.yticks([0.0, 1.0])


axins4 = inset_axes(ax2, width="50%", height="50%",
                    bbox_to_anchor=(.5, .15, .5, .5),
                    bbox_transform=ax2.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r2_ws_matrix, R2_ws_matrix,
                                     second_community_color,
                                     "#fdd0a2", alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins4, "$p_{out}$", "$\\langle R_2 \\rangle_t$",
                          x_lim_ws, y_lim_ws, (0.5, -0.2), (-0.25, 0.35),
                          inset_fontsize, inset_labelsize, linewidth)
plt.xticks([0.0, 1.0])
plt.yticks([0.0, 1.0])

plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the plot ?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    # file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    fig.savefig("data/winfree/"
                "{}_{}_complete_vs_reduced_winfree_2D"
                ".png".format(timestr, "without_perturbations"))  # file
    fig.savefig("data/winfree/"
                "{}_{}_complete_vs_reduced_winfree_2D"
                ".pdf".format(timestr, "without_perturbation"))  # file
