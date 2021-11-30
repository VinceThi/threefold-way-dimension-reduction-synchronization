import matplotlib.pyplot as plt
import itertools
import matplotlib.gridspec as gridspec
import numpy as np


def plot_PCA_reduction_matrices(X_transform,
                                principal_component_indices,
                                c1, c2, c3, fontsize, labelpad,
                                labelsize, xlim, ylim, xticks, yticks):
    """

    :param X_transform:
    :param principal_component_indices: Tuple (x, y) where x < y and x, y are
                                        in {0, 1, 2}.
    :param c1:
    :param c2:
    :param c3:
    :param fontsize:
    :param labelpad:
    :param labelsize:
    :param xlim:
    :param ylim:
    :param xticks:
    :param yticks:
    :return:
    """
    marker = itertools.cycle(('*', '^', 'P', 'o', 's'))
    markersize = itertools.cycle((140, 150, 150, 200, 200))
    zorder = itertools.cycle((13, 10, 7, 4, 1))
    color = 5 * [c1] + 5 * [c2] + 5 * [c3]
    x, y = principal_component_indices
    jj = 0
    for ii in range(len(X_transform[:, 0])):
        if not ii % 5 and ii:
            jj += 1
        plt.scatter(X_transform[ii, x],
                    X_transform[ii, y],
                    zorder=next(zorder)+jj,
                    s=next(markersize),
                    color=color[ii],
                    marker=next(marker),
                    edgecolors='#525252')
        # label=targets_possibilities[i],
    plt.vlines(0, ylim[0], ylim[1], linewidth=0.6, color='k', zorder=0)
    plt.plot(np.linspace(xlim[0], xlim[1], 2), [0, 0], linewidth=0.6,
             color='k', zorder=0)
    plt.xlabel(f"PC$_{x+1}$", fontsize=fontsize)
    # f"({'%.2f' % explained_variance_ratio[x]})"
    plt.ylabel(f"PC$_{y+1}$", fontsize=fontsize, labelpad=labelpad)
    # f"({'%.2f' % explained_variance_ratio[y]})",
    # plt.legend(bbox_to_anchor=(1.15, -0.2), fontsize=fontsize_legend, ncol=3)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # plt.tight_layout()


def plot_PCA_reduction_matrices_realizations(X_transform,
                                             principal_component_indices,
                                             c1, c2, c3, fontsize, labelpad,
                                             labelsize, xlim, ylim,
                                             xticks, yticks):
    """

    :param X_transform:
    :param principal_component_indices: Tuple (x, y) where x < y and x, y are
                                        in {0, 1, 2}.
    :param c1:
    :param c2:
    :param c3:
    :param fontsize:
    :param labelpad:
    :param labelsize:
    :param xlim:
    :param ylim:
    :param xticks:
    :param yticks:
    :return:
    """
    marker = "*"        # itertools.cycle(('*', '^', 'P', 'o', 's'))
    markersize = 140    # itertools.cycle((140, 150, 150, 200, 200))
    # zorder = itertools.cycle((13, 10, 7, 4, 1))
    color = c3          # 5 * [c1] + 5 * [c2] + 5 * [c3]
    x, y = principal_component_indices
    jj = 0
    for ii in range(len(X_transform[:, 0])):
        if not ii % 5 and ii:
            jj += 1
        plt.scatter(X_transform[ii, x],
                    X_transform[ii, y],
                    s=markersize,
                    color=color,
                    marker=marker,
                    edgecolors='#525252')
        # zorder=next(zorder)+jj,
        # label=targets_possibilities[i],
    plt.vlines(0, ylim[0], ylim[1], linewidth=0.6, color='k', zorder=0)
    plt.plot(np.linspace(xlim[0], xlim[1], 2), [0, 0], linewidth=0.6,
             color='k', zorder=0)
    plt.xlabel(f"PC$_{x+1}$", fontsize=fontsize)
    # f"({'%.2f' % explained_variance_ratio[x]})"
    plt.ylabel(f"PC$_{y+1}$", fontsize=fontsize, labelpad=labelpad)
    # f"({'%.2f' % explained_variance_ratio[y]})",
    # plt.legend(bbox_to_anchor=(1.15, -0.2), fontsize=fontsize_legend, ncol=3)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # plt.tight_layout()


def plot_PCA_components(ax, principal_component, n, N,
                        principal_component_index,
                        explained_variance_ratio, fontsize,
                        labelpad, labelsize, xlim, ylim):
    """

    :param ax:
    :param principal_component:
    :param n:
    :param N:
    :param principal_component_index:
    :param explained_variance_ratio:
    :param fontsize:
    :param labelpad:
    :param labelsize:
    :param xlim:
    :param ylim:
    :return:
    """
    principal_component_unflatten = np.reshape(principal_component, (n, N))
    # colors = 5*[c1] + 5*[c2] + 5*[c3]
    pci = principal_component_index
    matshow = ax.matshow(principal_component_unflatten,
                         aspect="auto", cmap=plt.cm.get_cmap("RdGy"))
    # x_arange = np.arange(0, len(principal_component))
    # x_arange_1 = np.arange(0, len(principal_component)+1)
    # ax.bar(x_arange, principal_component, color="#525252")
    # ax.plot(x_arange_1, np.zeros(len(x_arange_1)), "k", linewidth=0.2)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_ylabel(f"PC$_{pci+1}$", fontsize=fontsize, labelpad=labelpad)
    # ax.text(0, 0, f"EVR$_{pci+1}$ = "
    #         f"{'%.2f' % explained_variance_ratio[pci]}")
    ax.set_title(f"EVR$_{pci+1}$ = "
                 f"{'%.2f' % explained_variance_ratio[pci]}",
                 fontsize=fontsize, y=0.9)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return matshow


def plot_three_PCA_and_components(X_transform,
                                  principal_component_matrix, n, N,
                                  explained_variance_ratio):
    """
    :param X_transform:
    :param principal_component_matrix:
    :param n:
    :param N:
    :param explained_variance_ratio:
    :return:
    """

    c1 = "#9ecae1"
    c2 = "#fdd0a2"
    c3 = "#a1d99b"
    fontsize = 12
    # fontsize_legend = 12
    labelsize = 12
    # linewidth = 2
    labelpad = -6
    labelpad_PC = 0
    ylim_PC = None  # [-0.5, 0.75]
    xlim_PC = None  # [-0.5, 15]
    xlim = [-1.2, 1.2]
    ylim = [-1.2, 1.2]
    # xlim = [-0.1, 0.1]
    # ylim = [-0.02, 0.02]
    # xticks = [-0.1, 0, 0.1]
    # yticks = [-0.02, 0, 0.02]
    yticks = [-1.2, 0, 1.2]
    xticks = [-1.2, 0, 1.2]
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(5, 5))

    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    principal_component_indices_list = [(0, 1), (1, 2), (0, 2)]

    for i in range(4):
        if i == 3:
            inner = gridspec.GridSpecFromSubplotSpec(3, 1,
                                                     subplot_spec=outer[i],
                                                     wspace=0, hspace=0.8,
                                                     height_ratios=None,
                                                     width_ratios=None)
            for j in range(3):
                ax = plt.Subplot(fig, inner[j])
                matshow = plot_PCA_components(ax,
                                              principal_component_matrix[j],
                                              n, N, j,
                                              explained_variance_ratio,
                                              fontsize, labelpad_PC, labelsize,
                                              xlim_PC, ylim_PC)
                # if j < 2:
                ax.set_xticks([])
                ax.set_yticks([])
                box = ax.get_position()
                ax.set_position([box.x0*0.01, box.y0*-0.5,
                                 box.width, box.height])
                fig.colorbar(matshow, ax=ax, aspect=4, ticks=[-0.3, 0, 0.3])
                fig.add_subplot(ax)
        else:
            ax = plt.subplot(2, 2, i+1)
            plot_PCA_reduction_matrices(X_transform,
                                        principal_component_indices_list[i],
                                        c1, c2, c3, fontsize, labelpad,
                                        labelsize, xlim, ylim, xticks, yticks)

            fig.add_subplot(ax)

    plt.subplots_adjust(top=0.973,
                        bottom=0.1,
                        left=0.115,
                        right=0.958,
                        hspace=0.712,
                        wspace=0.466)
    # plt.tight_layout()
    plt.show()


def plot_three_PCA_and_components_realizations(X_transform,
                                               principal_component_matrix,
                                               n, N, explained_variance_ratio):
    """
    :param X_transform:
    :param principal_component_matrix:
    :param n:
    :param N:
    :param explained_variance_ratio:
    :return:
    """

    c1 = "#9ecae1"
    c2 = "#fdd0a2"
    c3 = "#a1d99b"
    fontsize = 12
    # fontsize_legend = 12
    labelsize = 12
    # linewidth = 2
    labelpad = -6
    labelpad_PC = 0
    ylim_PC = None  # [-0.5, 0.75]
    xlim_PC = None  # [-0.5, 15]
    # xlim = [-1.05, 1.05]
    # ylim = [-1.05, 1.05]
    xlim = [-0.2, 0.2]
    ylim = [-0.02, 0.02]
    xticks = [xlim[0], 0, xlim[-1]]
    yticks = [ylim[0], 0, ylim[-1]]
    # yticks = [-1, -0.5, 0, 0.5, 1]
    # xticks = [-1, -0.5, 0, 0.5, 1]
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(5, 5))

    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    principal_component_indices_list = [(0, 1), (1, 2), (0, 2)]

    for i in range(4):
        if i == 3:
            inner = gridspec.GridSpecFromSubplotSpec(3, 1,
                                                     subplot_spec=outer[i],
                                                     wspace=0, hspace=0.8,
                                                     height_ratios=None,
                                                     width_ratios=None)
            for j in range(3):
                ax = plt.Subplot(fig, inner[j])
                matshow = plot_PCA_components(ax,
                                              principal_component_matrix[j],
                                              n, N, j,
                                              explained_variance_ratio,
                                              fontsize, labelpad_PC, labelsize,
                                              xlim_PC, ylim_PC)
                # if j < 2:
                ax.set_xticks([])
                ax.set_yticks([])
                box = ax.get_position()
                ax.set_position([box.x0*0.01, box.y0*-0.5,
                                 box.width, box.height])
                fig.colorbar(matshow, ax=ax, aspect=4, ticks=[-0.3, 0, 0.3])
                fig.add_subplot(ax)
        else:
            ax = plt.subplot(2, 2, i+1)
            plot_PCA_reduction_matrices_realizations(
                X_transform, principal_component_indices_list[i],
                c1, c2, c3, fontsize, labelpad,
                labelsize, xlim, ylim, xticks, yticks)

            fig.add_subplot(ax)

    plt.subplots_adjust(top=0.973,
                        bottom=0.1,
                        left=0.130,
                        right=0.958,
                        hspace=0.712,
                        wspace=0.466)
    # plt.tight_layout()
    plt.show()
