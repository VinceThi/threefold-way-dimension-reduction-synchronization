import matplotlib.pyplot as plt
import numpy as np


def left_bottom_axis_settings(ax, xlabel, ylabel, xlim, ylim, x_coord, y_coord,
                              fontsize, labelsize, linewidth):
    plt.xlabel(xlabel, fontsize=fontsize)
    ylab = plt.ylabel(ylabel, fontsize=fontsize)
    ylab.set_rotation(0)
    ax.xaxis.set_label_coords(*x_coord)
    ax.yaxis.set_label_coords(*y_coord)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth-1)


# def plot_transitions_complete_vs_reduced(x_array, r_matrix, R_matrix,
#                                          complete_color, reduced_color,
#                                          alpha, marker, s, linewidth,
#                                          nb_instances):
#
#     # mean_r = np.mean(r_matrix, axis=0)
#     mean_R = np.mean(R_matrix, axis=0)
#     # std_r = np.std(r_matrix, axis=0)
#     std_R = np.std(R_matrix, axis=0)
#
#     for i in range(0, nb_instances):
#         if i == 0:
#             plt.scatter(x_array, r_matrix[i], color=complete_color, s=s,
#                         alpha=0.9, marker=marker)
#         else:
#             plt.scatter(x_array, r_matrix[i], color=complete_color, s=s,
#                         alpha=0.9, marker=marker)
#
#     plt.fill_between(x_array, mean_R - std_R, mean_R + std_R,
#                      color=reduced_color, alpha=alpha)
#     plt.plot(x_array, mean_R, color=reduced_color, linewidth=linewidth,
#              linestyle='-')


def plot_transitions_complete_vs_reduced(ax, x_array, r_matrix, R_matrix,
                                         complete_color, reduced_color,
                                         alpha, marker, s, linewidth,
                                         number_realizations):
    mean_r = np.mean(r_matrix, axis=0)
    mean_R = np.mean(R_matrix, axis=0)
    std_r = np.std(r_matrix, axis=0)
    std_R = np.std(R_matrix, axis=0)

    # zorder_complete = np.arange(0, number_realizations)
    # zorder_reduced = np.arange(number_realizations, 2*number_realizations)
    #
    # for i in range(0, number_realizations):
    #     ax.scatter(x_array, r_matrix[i], color=complete_color, s=s,
    #                alpha=0.9, marker=marker, zorder=zorder_complete[i])
    #     ax.scatter(x_array, R_matrix[i], color=reduced_color, s=s,
    #                alpha=0.9, marker=marker, zorder=zorder_reduced[i])
    plt.fill_between(x_array, mean_r - std_r, mean_r + std_r,
                     color=complete_color, alpha=alpha)
    ax.plot(x_array, mean_r, color=complete_color, linewidth=linewidth,
            linestyle='-')
    # # ax.fill_between(x_array, mean_r, mean_R,
    # #                 color=complete_color, alpha=0.2)
    plt.fill_between(x_array, mean_R - std_R, mean_R + std_R,
                     color=reduced_color, alpha=alpha)
    ax.plot(x_array, mean_R, color=reduced_color, linewidth=linewidth,
            linestyle='-')


def plot_transitions_complete_vs_reduced_one_instance(ax, sigma_array, r_array,
                                                      R_array, complete_color,
                                                      reduced_color, alpha,
                                                      marker, s, linewidth):
    # plt.scatter(x_array, r_array, color=complete_color, s=s,
    #             alpha=alpha, marker=marker)
    # plt.plot(x_array, R_array, color=reduced_color, linewidth=linewidth,
    #          linestyle='-')
    ax.scatter(sigma_array, r_array, color=complete_color,
               label="$\\langle R^{com} \\rangle_t$", s=s,
               alpha=alpha, marker=marker)
    # ax.scatter(sigma_array, R_array,
    #            color=reduced_color, linestyle="-",
    #            label="$\\langle R^{red} \\rangle_t$", s=s,
    #            alpha=alpha, marker=marker)
    # ax.plot(sigma_array, r_array, color=complete_color,
    #         label="$\\langle R^{com} \\rangle_t$", linewidth=linewidth)
    ax.plot(sigma_array, R_array,
            color=reduced_color, linestyle="-",
            label="$\\langle R^{red} \\rangle_t$", linewidth=linewidth)
    ax.fill_between(sigma_array, r_array, R_array,
                    color=complete_color, alpha=0.2)


def plot_multiple_transitions_complete_vs_reduced(ax, sigma_array, r_array,
                                                  R_array, complete_color,
                                                  reduced_color, linewidth):
    ax.plot(sigma_array, r_array, color=complete_color,
            label="$\\langle R^{com} \\rangle_t$", linewidth=linewidth)
    ax.plot(sigma_array, R_array,
            color=reduced_color, linestyle="-",
            label="$\\langle R^{red} \\rangle_t$", linewidth=linewidth)
    ax.fill_between(sigma_array, r_array, R_array,
                    color=complete_color, alpha=0.2)

# ax.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.xticks(xticks)
# ax.set_ylim([0, 1.02])
# ax.set_xlim(xlim)
# # ylab = plt.ylabel("$\\langle R \\rangle_t$",
# #                   fontsize=fontsize, labelpad=12)
# # ylab.set_rotation(0)
# plt.xlabel("$\\sigma$", fontsize=fontsize)
# plt.legend(loc=4, fontsize=fontsize_legend, handlelength=0.9)
# plt.tight_layout()
# plt.show()
