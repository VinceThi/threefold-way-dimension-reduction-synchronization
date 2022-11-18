import matplotlib.pyplot as plt
import matplotlib
import time as timer
import numpy as np
import networkx as nx
# from graphs.special_graphs import two_star_graph_adjacency_matrix
from scipy.sparse.linalg import eigs


def plot_complex_spectrum(network_generator, n_networks,
                          color="#064878",
                          markersize=10,
                          xlabel="Re[$\\lambda$]",
                          ylabel="Im[$\\lambda$]",
                          label_fontsize=13, labelsize=13,
                          sparse=False, nb_eigen=1000):

    t0 = timer.clock()
    eigenval = np.array([])
    # print(eigenval)
    i = 0
    for k in range(0, n_networks):
        A = network_generator
        if not sparse:
            val, vec = np.linalg.eig(A)
        else:
            val, vec = eigs(A, k=nb_eigen, which="LM")
        # print(val)
        eigenval = np.concatenate((eigenval, val))
        i += 1

    print((timer.clock()-t0)/60, "minutes to process")

    # print(eigenval)

    x = np.real(eigenval)
    y = np.imag(eigenval)
    # for i, v in enumerate(vals):
    #     c = axvline_color
    #     if isinstance(axvline_color, list):
    #         c = axvline_color[i]
    #     ax.axvline(v, color=c, zorder=0)

    # weights = np.ones_like(x) / float(len(x))

    ax = plt.gca()
    plt.scatter(x, y, s=markersize, marker="o", color=color)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    ax.xaxis.set_label_coords(1, 0.65)
    ylab = plt.ylabel(ylabel, fontsize=label_fontsize)  # , labelpad=10)
    ax.yaxis.set_label_coords(0.45, 1.05)
    ylab.set_rotation(0)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.xticks([np.round(min(x)), 0, np.round(max(x))])
    plt.yticks([np.round(min(y)), np.round(max(y))])
    
    plt.tight_layout()
    return


def plot_spectrum(ax, network_generator, n_networks, vals=[],
                  axvline_color="#ef8a62", bar_color="#064878",
                  normed=True, xlabel="$\\lambda$", ylabel="$\\rho(\\lambda)$",
                  label_fontsize=13, nbins=1000, labelsize=13):
    """

    :param ax:
    :param network_generator: A matrix
    :param n_networks:
    :param vals:
    :param axvline_color:
    :param bar_color:
    :param normed:
    :param xlabel:
    :param ylabel:
    :param label_fontsize:
    :param nbins:
    :param labelsize:
    :return:
    """
    # Edward and Vincent code
    # Get stats
    t0 = timer.clock()
    eigenval = np.array([])
    # print(eigenval)
    i = 0
    for k in range(0, n_networks):
        A = network_generator
        val, vec = np.linalg.eig(A)
        # print(val)
        eigenval = np.concatenate((eigenval, val))
        i += 1

    print((timer.clock()-t0)/60, "minutes to process")

    x = np.real(eigenval)
    # y = np.imag(eigenval)
    for i, v in enumerate(vals):
        c = axvline_color
        if isinstance(axvline_color, list):
            c = axvline_color[i]
        ax.axvline(v, color=c, zorder=0)

    weights = np.ones_like(x) / float(len(x))
    plt.hist(x, bins=nbins, color=bar_color, edgecolor="white",
             linewidth=1, weights=weights)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    ylab = plt.ylabel(ylabel, fontsize=label_fontsize, labelpad=20)
    ylab.set_rotation(0)
    plt.tight_layout()

    return


def plot_eigenvalue_density_SBM(n_networks, size, pq, ax, vals=[],
                                axvline_color="#ef8a62", bar_color="#064878",
                                normed=True, xlabel="$\\lambda$",
                                ylabel="$\\rho(\\lambda)$", label_fontsize=13,
                                nbins=1000, labelsize=13):

    # Get stats
    t0 = timer.clock()
    eigenval = np.array([])
    i = 0
    for k in range(0, n_networks):
        A = nx.to_numpy_matrix(nx.stochastic_block_model(size, pq))
        # K = np.diag(np.dot(A, np.ones(len(A[:, 0])).transpose()))
        # L = K-A
        val, vec = np.linalg.eig(A)
        eigenval = np.concatenate((eigenval, val))
        i += 1
    print((timer.clock() - t0) / 60, "minutes to process")

    x = np.real(eigenval)
    # y = np.imag(eigenval)
    for i, v in enumerate(vals):
        c = axvline_color
        if isinstance(axvline_color, list):
            c = axvline_color[i]
        ax.axvline(v, color=c, zorder=0)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True
    n, bins, patches = plt.hist(x, bins=nbins, color=bar_color,
                                edgecolor="#B2B2B2", linewidth=3,
                                density=normed)
    plt.plot(bins[:-1], n, linewidth=2, color='k')
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize, labelpad=20)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    q = 2
    sizes = [120, 80]         # 50/30, 100/60, 200/120
    N = sum(sizes)
    n1 = sizes[0]
    n2 = sizes[1]
    plt.figure(figsize=(3, 6))
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True

    nbins = 30

    ax1 = plt.subplot(411)
    A_erdos = nx.to_numpy_matrix(nx.stochastic_block_model(sizes, [[0.5, 0.5],
                                                                   [0.5, 0.5]]
                                                           ))
    plot_spectrum(ax1, A_erdos, 1, vals=[],
                  axvline_color="#2171b5", bar_color="#b2b2b2",
                  normed=True, xlabel="",
                  ylabel="",
                  label_fontsize=13, nbins=nbins, labelsize=13)
    # plt.xlim([-2, 3])

    ax2 = plt.subplot(412)
    A_bip = nx.to_numpy_matrix(nx.stochastic_block_model(sizes, [[0, 0.5],
                                                                 [0.5, 0]]))
    plot_spectrum(ax2, A_bip, 1, vals=[],
                  axvline_color="#2171b5", bar_color="#b2b2b2",
                  normed=True, xlabel="",
                  ylabel="",
                  label_fontsize=13, nbins=nbins, labelsize=13)
    # plt.xlim([-2, 3])

    ax3 = plt.subplot(413)
    A_SBM = nx.to_numpy_matrix(nx.stochastic_block_model(sizes, [[0.8, 0.2],
                                                                 [0.2, 0.8]]))
    plot_spectrum(ax3, A_SBM, 1, vals=[],
                  axvline_color="#2171b5", bar_color="#b2b2b2",
                  normed=True, xlabel="", ylabel="",
                  label_fontsize=13, nbins=nbins, labelsize=13)
    # plt.xlim([-2, 3])

    ax4 = plt.subplot(414)
    # A_TS = two_star_graph_adjacency_matrix(sizes, [[1, 1], [1, 1]])
    # plot_spectrum(ax4, A_TS, 1, vals=[],
    #               axvline_color="#2171b5", bar_color="#b2b2b2",
    #               normed=True, xlabel="$\\lambda$", ylabel="",
    #               label_fontsize=13, nbins=nbins, labelsize=13)
    # # plt.xlim([-15, 15])

    plt.tight_layout()

    plt.show()
    # plt.figure(figsize=(3, 3))
    # matplotlib.rcParams['text.usetex'] = True
    # matplotlib.rcParams['text.latex.unicode'] = True
    # ax1 = plt.subplot(211)
    # A = np.array([[0, 1, 1, 0, 0, 0],
    #               [1, 0, 1, 0, 0, 0],
    #               [1, 1, 0, 1, 0, 0],
    #               [0, 0, 1, 0, 1, 1],
    #               [0, 0, 0, 1, 0, 1],
    #               [0, 0, 0, 1, 1, 0]])
    # plot_spectrum(ax1, A, 1, vals=[],
    #               axvline_color="#2171b5", bar_color="#2171b5",
    #               normed=True, xlabel="", ylabel="$\\rho_A(\\lambda)$",
    #               label_fontsize=13, nbins=24, labelsize=13)
    # plt.xlim([-2, 3])
    #
    # ax2 = plt.subplot(212)
    # from numpy.linalg import pinv, multi_dot
    # M = np.array([[0.2929, 0.2929, 0.3143, 0.0999, 0, 0],
    #               [0, 0, 0.0999, 0.3143, 0.2929, 0.2929]])
    # redA = multi_dot([M, A, pinv(M)])
    # plot_spectrum(ax2, redA, 1, vals=[],
    #               axvline_color="#2171b5", bar_color="#2171b5",
    #               normed=True, xlabel="$\\lambda$",
    #               ylabel="$\\rho_{\\mathcal{A}}(\\lambda)$",
    #               label_fontsize=13, nbins=4, labelsize=13)
    # plt.xlim([-2, 3])
    #
    # plt.show()

    # Plot SBM
    # Structural parameter of the SBM
    # from synch_predictions.graphs.SBM_geometry import give_pq
    # from scipy.linalg import eig
    # q = 2
    # sizes = [120, 80]         # 50/30, 100/60, 200/120
    # N = sum(sizes)
    # n1 = sizes[0]
    # n2 = sizes[1]
    # f1 = n1/N
    # f2 = n2/N
    # f = f1/f2
    # beta = (n1*(n1-1) + n2*(n2-1)) / (N*(N-1))
    # rho = 0.535306122449
    # delta_indet = 0.01
    # pq_indet = give_pq(rho, delta_indet, beta)
    # A = nx.to_numpy_matrix(nx.stochastic_block_model(sizes, pq_indet))
    # eigvapvep = eig(A)
    # VAPs = eigvapvep[0]
    # VAPs_indet = sorted(np.absolute(VAPs))[::-1]
    #
    # delta_det = 0.8
    # pq_det = give_pq(rho, delta_det, beta)
    # A = nx.to_numpy_matrix(nx.stochastic_block_model(sizes, pq_det))
    # eigvapvep = eig(A)
    # VAPs = eigvapvep[0]
    # VAPs_det = sorted(np.absolute(VAPs))[::-1]
    #
    #
    # ax = plt.subplot(111)
    # # ax.vlines(VAPs_indet[0], 0, 0.01,
    # #           color=first_community_color, linewidth=4)
    # # ax.vlines(VAPs_indet[1], 0, 0.01,
    # #           color=second_community_color, linewidth=4)
    #
    # plot_eigenvalue_density_SBM(1, sizes, pq_det, ax, vals=[],
    #                             axvline_color="#ef8a62", bar_color="#525252",
    #                             normed=True, xlabel="$\\lambda$",
    #                             ylabel="$\\rho(\\lambda)$", label_fontsize=0,
    #                             nbins=500, labelsize=20)
    # plt.show()

    #
    # #ax = plt.subplot(111)
    # ##ax.vlines(VAPs_det[0], 0, 0.05,
    #  color=first_community_color, linewidth=4)
    # ##ax.vlines(VAPs_det[1], 0, 0.05,
    #  color=second_community_color, linewidth=4)
    # #plot_eigenvalue_density_SBM(2000, sizes, pq_det, ax, vals=[],
    #  axvline_color="#ef8a62",bar_color="#B2B2B2",normed=True,
    # xlabel="$\\lambda$",ylabel="$\\rho(\\lambda)$",label_fontsize=0,
    # nbins=1000, labelsize=20)
