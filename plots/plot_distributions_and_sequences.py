import matplotlib.pyplot as plt
from synch_predictions.graphs.get_distributions_and_sequences import *


def plot_distribution(ax, array, width=1, color="#fdd0a2",
                      xlabel="", ylabel="",
                      xy_log_scale=False):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True
    deg, cnt = get_distribution(array)
    ax.bar(deg, np.array(cnt)/np.sum(np.array(cnt)), width=width, color=color)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if xy_log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.tight_layout()


def plot_sequence(ax, array, width=0.5, color="#fdd0a2",
                  xlabel="", ylabel="", xticks=None,
                  yticks=None, labelpad=None, labelsize=12, fontsize=12,
                  xy_log_scale=False):
    # xticks = (1,2,3,4,5,6) in DART paper, it is a tuple.
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True
    ax.bar(np.arange(1, len(array)+1), array,
           width=width, color=color)
    ylab = ax.set_ylabel(ylabel, labelpad=labelpad, fontsize=fontsize)
    # ylab.set_rotation(0)
    ax.set_xlabel(xlabel)
    if xticks:
        ax.set_xticks(list(xticks))
    else:
        ax.set_xticks([])
    if type(yticks) == tuple:
        ax.set_yticks(list(yticks))
    if xy_log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tight_layout()



def plot_degree_distribution(ax, G, width=1, color="#fdd0a2",
                             xy_log_scale=False):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True
    deg, cnt = get_degree_distribution(G)
    plt.bar(deg, np.array(cnt)/np.sum(np.array(cnt)), width=width, color=color)
    plt.ylabel("$P(k)$")
    plt.xlabel("Degree $k$")
    if xy_log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.tight_layout()


def plot_degree_sequence(ax, G, width=0.5, color="#fdd0a2",
                         xy_log_scale=False):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True
    degree_sequence = get_degree_sequence(G)
    ax.bar(np.arange(0, len(degree_sequence)), degree_sequence,
            width=width, color=color)
    ax.ylabel("$k_j$")
    ax.xlabel("Indices $j$")
    if xy_log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.tight_layout()
