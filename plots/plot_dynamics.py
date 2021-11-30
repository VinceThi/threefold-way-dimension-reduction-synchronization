# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


from synch_predictions.plots.plots_setup import *


def plot_dynamics_vs_time(time_list, temporal_series, series_label="",
                          color="k", linestyle="-", legend=0):
    plt.plot(time_list, temporal_series, linestyle=linestyle,
             linewidth=linewidth, color=color, label=series_label)
    plt.xlabel("$t$", fontsize=fontsize)
    if legend == 1:
        plt.legend(loc="best", fontsize=fontsize_legend)
    else:
        plt.ylabel(series_label, fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tight_layout()
    # plt.ylim([0, 1.01])
    # ax = plt.subplot(111)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    # plt.show()
