from synch_predictions.plots.plot_complete_vs_reduced import *
from synch_predictions.simulations.data_synchro_transition_winfree import *
from synch_predictions.simulations.data_synchro_transition_kuramoto import *
from synch_predictions.simulations.data_synchro_transition_theta import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Not the typical setups for the fontsize
first_community_color = "#2171b5"
reduced_first_community_color = "#9ecae1"
second_community_color = "#f16913"
reduced_second_community_color = "#fdd0a2"
fontsize = 14
inset_fontsize = 9
fontsize_legend = 14
labelsize = 14
inset_labelsize = 9
linewidth = 2
s = 20
alpha = 0.5
marker = "."
x_lim_wb = [0, 1.02]
y_lim_wb = [0, 1.02]
x_lim_ws = [0, 1.02]
y_lim_ws = [0, 1.02]
x_lim_kb = [0.08, 0.92]
y_lim_kb = [0, 1.02]
x_lim_ks = [0.08, 0.92]
y_lim_ks = [0, 1.02]
x_lim_te = [0, 1.02]
y_lim_te = [0, 1.02]
x_lim_ts = [0, 1.02]
y_lim_ts = [0, 1.02]
nb_instances = 50

# Horizontal (one column) plot
fig = plt.figure(figsize=(10, 5))

# Winfree on bipartite
ax = plt.subplot(231)
# ax.title.set_text('$(a)$')
# ax.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r_wb_matrix, R_wb_matrix,
                                     "#252525", "#969696",
                                     alpha, marker, s, linewidth, nb_instances)
left_bottom_axis_settings(ax, "$p_{out}$", "$\\langle R\, \\rangle_t$",
                          x_lim_wb, y_lim_wb,
                          (0.5, -0.15), (-0.3, 0.45), fontsize,
                          labelsize, linewidth)

axins1 = inset_axes(ax, width="50%", height="50%",
                    bbox_to_anchor=(-0.08, .7, .5, .5),   # (.5, .55, .5, .5),
                    bbox_transform=ax.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r1_wb_matrix, R1_wb_matrix,
                                     first_community_color,
                                     "#9ecae1", alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins1, "$p_{out}$", "$\\langle R_1\\rangle_t$",
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

ax2 = plt.subplot(234)
# ax2.title.set_text('$(b)$')
# ax2.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r_ws_matrix, R_ws_matrix,
                                     "#252525", "#969696",
                                     alpha, marker, s, linewidth, nb_instances)
left_bottom_axis_settings(ax2, "$p_{out}$", "$\\langle R\, \\rangle_t$",
                          x_lim_ws, y_lim_ws,
                          (0.5, -0.15), (-0.3, 0.45), fontsize,
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



# Kuramoto on bipartite
ax = plt.subplot(232)
# ax.title.set_text('$(c)$')
# ax.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(np.linspace(0.1, 0.9, 50),
                                     r_kb_matrix, R_kb_matrix,
                                     "#252525", "#969696",
                                     alpha, marker, s, linewidth, nb_instances)
left_bottom_axis_settings(ax, "$p_{out}$", "$\\langle R\, \\rangle_t$",
                          x_lim_kb, y_lim_kb,
                          (0.5, -0.15), (-0.3, 0.45), fontsize,
                          labelsize, linewidth)
plt.xticks([0.1, 0.5, 0.9])

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

ax2 = plt.subplot(235)
# ax2.title.set_text('$(d)$')
# ax2.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r_ks_matrix, R_ks_matrix,
                                     "#252525", "#969696",
                                     alpha, marker, s, linewidth, nb_instances)
left_bottom_axis_settings(ax2, "$p_{out}$", "$\\langle R\, \\rangle_t$",
                          x_lim_ks, y_lim_ks,
                          (0.5, -0.15), (-0.3, 0.45), fontsize,
                          labelsize, linewidth)
plt.xticks([0.1, 0.5, 0.9])


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

# theta on Erdos-Renyi graph
ax = plt.subplot(233)
# ax.title.set_text('$(e)$')
# ax.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r_I1_matrix, R_I1_matrix,
                                     "#252525", "#969696",
                                     alpha, marker, s, linewidth, nb_instances)
left_bottom_axis_settings(ax, "$p$", "$\\langle R\, \\rangle_t$",
                          x_lim_te, y_lim_te,
                          (0.5, -0.15), (-0.3, 0.45), fontsize,
                          labelsize, linewidth)


# theta on SBM
ax2 = plt.subplot(236)
# ax2.title.set_text('$(f)$')
# ax2.title.set_position((0.5, 1.05))
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r_ts_matrix, R_ts_matrix,
                                     "#252525", "#969696",
                                     alpha, marker, s, linewidth, nb_instances)
left_bottom_axis_settings(ax2, "$p_{out}$", "$\\langle R\, \\rangle_t$",
                          x_lim_ts, y_lim_ts,
                          (0.5, -0.15), (-0.3, 0.45), fontsize,
                          labelsize, linewidth)


axins3 = inset_axes(ax2, width="50%", height="50%",
                    bbox_to_anchor=(-0.09, .6, .5, .5),   # (.5, .55, .5, .5),
                    bbox_transform=ax2.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r1_ts_matrix, R1_ts_matrix,
                                     first_community_color,
                                     "#9ecae1", alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins3, "$p_{out}$", "$\\langle R_1 \\rangle_t$",
                          x_lim_ts, y_lim_ts, (0.5, -0.2), (-0.25, 0.35),
                          inset_fontsize, inset_labelsize, linewidth)
plt.xticks([0.0, 1.0])
plt.yticks([0.0, 1.0])


axins4 = inset_axes(ax2, width="50%", height="50%",
                    bbox_to_anchor=(.45, .6, .5, .5),  # (.5, .15, .5, .5),
                    bbox_transform=ax2.transAxes, loc=4)
plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
                                     r2_ts_matrix, R2_ts_matrix,
                                     second_community_color,
                                     "#fdd0a2", alpha,
                                     marker, s, linewidth, nb_instances)
left_bottom_axis_settings(axins4, "$p_{out}$", "$\\langle R_2 \\rangle_t$",
                          x_lim_ts, y_lim_ts, (0.5, -0.2), (-0.25, 0.35),
                          inset_fontsize, inset_labelsize, linewidth)
plt.xticks([0.0, 1.0])
plt.yticks([0.0, 1.0])

plt.tight_layout()

plt.show()

# Vertical (one column) plot
# fig = plt.figure(figsize=(7, 9))
#
# # Winfree on SBM graph
# ax = plt.subplot(321)
# # ax.title.set_text('$(a)$')
# # ax.title.set_position((0.5, 1.05))
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r_wb_matrix, R_wb_matrix,
#                                      "#252525", "#969696",
#                                      alpha, marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(ax, "$p_{out}$", "$\\langle R\, \\rangle_t$",
#                           x_lim_wb, y_lim_wb,
#                           (0.5, -0.15), (-0.3, 0.45), fontsize,
#                           labelsize, linewidth)
#
# axins1 = inset_axes(ax, width="50%", height="50%",
#                     bbox_to_anchor=(-0.08, .7, .5, .5),   # (.5, .55, .5, .5),
#                     bbox_transform=ax.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r1_wb_matrix, R1_wb_matrix,
#                                      first_community_color,
#                                      "#9ecae1", alpha,
#                                      marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(axins1, "$p_{out}$", "$\\langle R_1\\rangle_t$",
#                           x_lim_wb, y_lim_wb, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0, 1])
#
#
# axins2 = inset_axes(ax, width="50%", height="50%",
#                     bbox_to_anchor=(.5, .15, .5, .5),
#                     bbox_transform=ax.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r2_wb_matrix, R1_wb_matrix,
#                                      second_community_color,
#                                      "#fdd0a2", alpha,
#                                      marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(axins2, "$p_{out}$", "$\\langle R_2 \\rangle_t$",
#                           x_lim_wb, y_lim_wb, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0, 1])
#
#
# # Winfree on SBM
#
# ax2 = plt.subplot(322)
# # ax2.title.set_text('$(b)$')
# # ax2.title.set_position((0.5, 1.05))
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r_ws_matrix, R_ws_matrix,
#                                      "#252525", "#969696",
#                                      alpha, marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(ax2, "$p_{out}$", "$\\langle R\, \\rangle_t$",
#                           x_lim_ws, y_lim_ws,
#                           (0.5, -0.15), (-0.3, 0.45), fontsize,
#                           labelsize, linewidth)
#
#
# axins3 = inset_axes(ax2, width="50%", height="50%",
#                     bbox_to_anchor=(-0.08, .15, .5, .5),
#                     bbox_transform=ax2.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r1_ws_matrix, R1_ws_matrix,
#                                      first_community_color,
#                                      "#9ecae1", alpha,
#                                      marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(axins3, "$p_{out}$", "$\\langle R_1 \\rangle_t$",
#                           x_lim_ws, y_lim_ws, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0.0, 1.0])
# plt.yticks([0.0, 1.0])
#
#
# axins4 = inset_axes(ax2, width="50%", height="50%",
#                     bbox_to_anchor=(.5, .15, .5, .5),
#                     bbox_transform=ax2.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r2_ws_matrix, R2_ws_matrix,
#                                      second_community_color,
#                                      "#fdd0a2", alpha,
#                                      marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(axins4, "$p_{out}$", "$\\langle R_2 \\rangle_t$",
#                           x_lim_ws, y_lim_ws, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0.0, 1.0])
# plt.yticks([0.0, 1.0])
#
# # Kuramoto on SBM graph
# ax = plt.subplot(323)
# # ax.title.set_text('$(c)$')
# # ax.title.set_position((0.5, 1.05))
# plot_transitions_complete_vs_reduced(np.linspace(0.1, 0.9, 50),
#                                      r_kb_matrix, R_kb_matrix,
#                                      "#252525", "#969696",
#                                      alpha, marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(ax, "$p_{out}$", "$\\langle R\, \\rangle_t$",
#                           x_lim_kb, y_lim_kb,
#                           (0.5, -0.15), (-0.3, 0.45), fontsize,
#                           labelsize, linewidth)
# plt.xticks([0.1, 0.5, 0.9])
#
# axins1 = inset_axes(ax, width="50%", height="50%",
#                     bbox_to_anchor=(.45, .55, .5, .5),   # (-0.12, .7, .5, .5),
#                     bbox_transform=ax.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r1_kb_matrix, R1_kb_matrix,
#                                      first_community_color,
#                                      reduced_first_community_color, alpha,
#                                      marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(axins1, "$p_{out}$", "$\\langle R_1 \\rangle_t$",
#                           x_lim_kb, y_lim_kb, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0.1, 0.9])
#
#
# axins2 = inset_axes(ax, width="50%", height="50%",
#                     bbox_to_anchor=(.45, .12, .5, .5),  # (.5, .15, .5, .5),
#                     bbox_transform=ax.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r2_kb_matrix, R1_kb_matrix,
#                                      second_community_color,
#                                      reduced_second_community_color, alpha,
#                                      marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(axins2, "$p_{out}$", "$\\langle R_2 \\rangle_t$",
#                           x_lim_kb, y_lim_kb, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0.1, 0.9])
#
#
# # Kuramoto on SBM
#
# ax2 = plt.subplot(324)
# # ax2.title.set_text('$(d)$')
# # ax2.title.set_position((0.5, 1.05))
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r_ks_matrix, R_ks_matrix,
#                                      "#252525", "#969696",
#                                      alpha, marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(ax2, "$p_{out}$", "$\\langle R\, \\rangle_t$",
#                           x_lim_ks, y_lim_ks,
#                           (0.5, -0.15), (-0.3, 0.45), fontsize,
#                           labelsize, linewidth)
# plt.xticks([0.1, 0.5, 0.9])
#
#
# axins3 = inset_axes(ax2, width="50%", height="50%",
#                     bbox_to_anchor=(-0.08, .15, .5, .5),   # (.5, .55, .5, .5),
#                     bbox_transform=ax2.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r1_ks_matrix, R1_ks_matrix,
#                                      first_community_color,
#                                      reduced_first_community_color, alpha,
#                                      marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(axins3, "$p_{out}$", "$\\langle R_1 \\rangle_t$",
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
#                                      marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(axins4, "$p_{out}$", "$\\langle R_2 \\rangle_t$",
#                           x_lim_ks, y_lim_ks, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0.1, 0.9])
# plt.yticks([0.0, 1.0])
#
# # theta on Erdos-Renyi graph
# ax = plt.subplot(325)
# # ax.title.set_text('$(e)$')
# # ax.title.set_position((0.5, 1.05))
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r_I1_matrix, R_I1_matrix,
#                                      "#252525", "#969696",
#                                      alpha, marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(ax, "$p$", "$\\langle R\, \\rangle_t$",
#                           x_lim_te, y_lim_te,
#                           (0.5, -0.15), (-0.3, 0.45), fontsize,
#                           labelsize, linewidth)
#
#
# # theta on SBM
# ax2 = plt.subplot(326)
# # ax2.title.set_text('$(f)$')
# # ax2.title.set_position((0.5, 1.05))
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r_ts_matrix, R_ts_matrix,
#                                      "#252525", "#969696",
#                                      alpha, marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(ax2, "$p_{out}$", "$\\langle R\, \\rangle_t$",
#                           x_lim_ts, y_lim_ts,
#                           (0.5, -0.15), (-0.3, 0.45), fontsize,
#                           labelsize, linewidth)
#
#
# axins3 = inset_axes(ax2, width="50%", height="50%",
#                     bbox_to_anchor=(-0.09, .6, .5, .5),   # (.5, .55, .5, .5),
#                     bbox_transform=ax2.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r1_ts_matrix, R1_ts_matrix,
#                                      first_community_color,
#                                      "#9ecae1", alpha,
#                                      marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(axins3, "$p_{out}$", "$\\langle R_1 \\rangle_t$",
#                           x_lim_ts, y_lim_ts, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0.0, 1.0])
# plt.yticks([0.0, 1.0])
#
#
# axins4 = inset_axes(ax2, width="50%", height="50%",
#                     bbox_to_anchor=(.45, .6, .5, .5),  # (.5, .15, .5, .5),
#                     bbox_transform=ax2.transAxes, loc=4)
# plot_transitions_complete_vs_reduced(np.linspace(0.01, 1, 50),
#                                      r2_ts_matrix, R2_ts_matrix,
#                                      second_community_color,
#                                      "#fdd0a2", alpha,
#                                      marker, s, linewidth, nb_instances)
# left_bottom_axis_settings(axins4, "$p_{out}$", "$\\langle R_2 \\rangle_t$",
#                           x_lim_ts, y_lim_ts, (0.5, -0.2), (-0.25, 0.35),
#                           inset_fontsize, inset_labelsize, linewidth)
# plt.xticks([0.0, 1.0])
# plt.yticks([0.0, 1.0])
#
# plt.tight_layout()
#
# plt.show()