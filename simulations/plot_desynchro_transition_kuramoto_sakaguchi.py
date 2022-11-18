from plots.plot_complete_vs_reduced import *
from simulations.data_synchro_transition_kuramoto_sakaguchi\
    import *
import matplotlib.pyplot as plt
import numpy as np
import tkinter.simpledialog
from tkinter import messagebox
import time
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
x_lim = None
y_lim = [0, 1.02]
nb_instances = 1
alpha_list = np.linspace(0.6, 1.2, 100)


def give_alpha_critical(alphal, Rpi_reduced_vs_alpha):
    """

    :param Rpi_reduced_vs_alpha: list of mean Rpi (i=1 or 2 (1st star or 2nd),
                                                   p for periphery,
                                                   R for the synchronization
                                                   parameter)
    :param alphal: The list of alpha values associated to Rpi_reduced_vs_alpha
    :return: critical_alpha
    """
    # First method
    i = 0
    tolerance = 0.001
    critical_alpha = None
    for Rp1 in Rpi_reduced_vs_alpha:
        if np.abs(1-Rp1) < tolerance:
            i += 1
        else:
            critical_alpha = (alphal[i] + alphal[i-1])/2

    # Second method
    delta_alpha = alphal[1]-alphal[0]
    derivative_dRpidalpha = np.diff(Rpi_reduced_vs_alpha)/delta_alpha
    index_min_value_derivative = np.argmin(derivative_dRpidalpha)
    critical_alpha_derivative = (alphal[index_min_value_derivative]
                                 + alphal[index_min_value_derivative-1])/2

    # We have two critical values obtained with two different methods,
    # are they similar ? (because they must be)
    tol = 0.05
    if np.abs(critical_alpha - critical_alpha_derivative) > tol:
        print("Error in function give_alpha_critical:"
              " The values of the critical alpha's must close"
              " to each other with the two methods.")

    return critical_alpha


fig = plt.figure(figsize=(6, 3.2))


# Kuramoto-Sakaguchi on two-star graph
ax = plt.subplot(121)
ax.title.set_text('$(a)$')
ax.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(alpha_list,
                                     [rp1_1], [Rp1_1],
                                     first_community_color,
                                     reduced_first_community_color, alpha,
                                     marker, s, linewidth, nb_instances)
plot_transitions_complete_vs_reduced(alpha_list,
                                     [rp2_1], [Rp2_1],
                                     second_community_color,
                                     reduced_second_community_color, alpha,
                                     marker, s, linewidth, nb_instances)
plt.axvline(give_alpha_critical(alpha_list, Rp1_1),
            linestyle='--', color="#404040", linewidth=1)
left_bottom_axis_settings(ax, "$\\alpha$", "$R_{\\mu}$", x_lim, y_lim,
                          (0.5, -0.15), (-0.2, 0.45), fontsize,
                          labelsize, linewidth)

ax2 = plt.subplot(122)
ax2.title.set_text('$(b)$')
ax2.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(alpha_list,
                                     [rp1_2], [Rp1_2],
                                     first_community_color,
                                     reduced_first_community_color, alpha,
                                     marker, s, linewidth, nb_instances)
plot_transitions_complete_vs_reduced(alpha_list,
                                     [rp2_2], [Rp2_2],
                                     second_community_color,
                                     reduced_second_community_color, alpha,
                                     marker, s, linewidth, nb_instances)
plt.axvline(give_alpha_critical(alpha_list, Rp1_2),
            linestyle='--', color="#404040", linewidth=1)
plt.axvline(give_alpha_critical(alpha_list, Rp2_2),
            linestyle='--', color="#404040", linewidth=1)
# plt.fill_between(alpha_list, mean_R - std_R, mean_R + std_R,
#                  color=reduced_first_community_color, alpha=alpha)
ax2.axvspan(give_alpha_critical(alpha_list, Rp1_2),
            give_alpha_critical(alpha_list, Rp2_2),
            alpha=0.5, color=reduced_second_community_color)

left_bottom_axis_settings(ax2, "$\\alpha$", "$R_{\\mu}$", x_lim, y_lim,
                          (0.5, -0.15), (-0.2, 0.45), fontsize,
                          labelsize, linewidth)

# plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])
# 
# 
# 
# axins2 = inset_axes(ax, width="50%", height="50%",
#                     bbox_to_anchor=(.45, .12, .5, .5),  # (.5, .15, .5, .5),
#                     bbox_transform=ax.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r2_kb_matrix, R1_kb_matrix,
#                                      second_community_color,
#                                      reduced_second_community_color, alpha,
#                                      marker, s, linewidth)
# left_bottom_axis_settings(axins2, "$p_{out}$", "$R_2$",
#                           x_lim_kb, y_lim_kb, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0.1, 0.9])
# 
# 
# # Kuramoto on SBM
# 
# ax2 = plt.subplot(122)
# ax2.title.set_text('$(b)$')
# ax2.title.set_position((0.5, 1.05))
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r_ks_matrix, R_ks_matrix,
#                                      "#252525", "#969696",
#                                      alpha, marker, s, linewidth)
# left_bottom_axis_settings(ax2, "$p_{out}$", "$R$", x_lim_ks, y_lim_ks,
#                           (0.5, -0.15), (-0.2, 0.45), fontsize,
#                           labelsize, linewidth)
# plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])
# 
# 
# axins3 = inset_axes(ax2, width="50%", height="50%",
#                     bbox_to_anchor=(-0.08, .15, .5, .5), # (.5, .55, .5, .5),
#                     bbox_transform=ax2.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r1_ks_matrix, R1_ks_matrix,
#                                      first_community_color,
#                                      reduced_first_community_color, alpha,
#                                      marker, s, linewidth)
# left_bottom_axis_settings(axins3, "$p_{out}$", "$R_1$",
#                           x_lim_ks, y_lim_ks, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0.1, 0.9])
# plt.yticks([0.0, 1.0])
# 
# 
# axins4 = inset_axes(ax2, width="50%", height="50%",
#                     bbox_to_anchor=(.45, .15, .5, .5),  # (.5, .15, .5, .5),
#                     bbox_transform=ax2.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r2_ks_matrix, R2_ks_matrix,
#                                      second_community_color,
#                                      reduced_second_community_color, alpha,
#                                      marker, s, linewidth)
# left_bottom_axis_settings(axins4, "$p_{out}$", "$R_2$",
#                           x_lim_ks, y_lim_ks, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0.1, 0.9])
# plt.yticks([0.0, 1.0])

plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the plot ?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    fig.savefig("data/kuramoto_sakaguchi/"
                "{}_{}_complete_vs_reduced_kuramoto_sakaguchi"
                ".png".format(timestr, file))
    fig.savefig("data/kuramoto_sakaguchi/"
                "{}_{}_complete_vs_reduced_kuramoto_sakaguchi"
                ".pdf".format(timestr, file))
