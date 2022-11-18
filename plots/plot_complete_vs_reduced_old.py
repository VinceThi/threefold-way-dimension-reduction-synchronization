# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


from plots.plots_setup import *


def plot_complete_vs_reduced_vs_time(time_list, r1, r2, R1, R2):
    plt.plot(time_list, r1, linestyle="-", linewidth=linewidth,
             color=first_community_color)  # , label="$r_{1}$")
    plt.plot(time_list, r2, linestyle="-", linewidth=linewidth,
             color=second_community_color)  # , label="$r_{2}$")
    plt.plot(time_list, R1, linestyle="--", linewidth=linewidth,
             color="#a8ddb5")  # , label="$R_{1}$")
    plt.plot(time_list, R2, linestyle="--", linewidth=linewidth,
             color="#feb24c")  # , label="$R_{2}$")
    plt.xlabel("$t$", fontsize=fontsize)
    #plt.legend(loc="best", fontsize=fontsize_legend, ncol=2)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tight_layout()
    # plt.ylim([0, 1.05])
    # ax = plt.subplot(111)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')


def plot_complete_vs_reduced1D_vs_time(time_list, r, R,
                                       color_r=first_community_color,
                                       color_R="#a8ddb5"):
    plt.plot(time_list, r, linestyle="-", linewidth=linewidth,
             color=color_r, label="$Complete$")
    plt.plot(time_list, R, linestyle="--", linewidth=linewidth,
             color=color_R, label="$Reduced$")
    plt.xlabel("$t$", fontsize=fontsize)
    plt.legend(loc="best", fontsize=fontsize_legend)#, ncol=2)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tight_layout()
    # plt.ylim([0, 1.05])
    # ax = plt.subplot(111)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')


def plot_complete_vs_reduced_vs_delta(delta_array, r1, r2, R1, R2,
                                      ymin=0, ymax=1, xmin=0, xmax=1):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True
    plt.plot(delta_array, r1, linewidth=linewidth,
             color=first_community_color, label="$r_{1}$")
    plt.plot(delta_array, r2, linewidth=linewidth,
             color=second_community_color, label="$r_{2}$")
    plt.plot(delta_array, R1, linestyle="--",
             linewidth=linewidth, color="#a8ddb5", label="$R_{1}$")
    plt.plot(delta_array, R2, linestyle="--",
             linewidth=linewidth, color="#feb24c", label="$R_{2}$")
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel("$\\Delta = p_{in} - p_{out}$", fontsize=fontsize)
    plt.legend(loc="best", fontsize=fontsize_legend, ncol=2)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tight_layout()
    # ax = plt.subplot(111)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
