import dynamicalab.drawing as draw
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


first_community_color = "#2171b5"
second_community_color = "#f16913"
third_community_color = "#6a51a3"
fourth_community_color = "#ef3b2c"
mu = 0
edge_color = "gray"
edge_width = 1.5
edge_alpha = 0.6
use_edge_weigth = False
node_width = 1.5
node_size = 50.0
reduced_node_size = 100.0
node_border_color = "#404040"
node_alpha = 1.0
arrow_scale = 0.0
loop_radius = 0


def draw_sbm_intro_article_synchro(sizes, pq, centroids, radius_scale):
    # Draw SBM

    ax_SBM = plt.subplot(111)
    n1, n2, n3, n4 = sizes
    G_SBM = nx.stochastic_block_model(sizes, pq)
    # pos = nx.layout.spectral_layout(G_SBM)
    # pos = nx.spring_layout(G_SBM, pos=pos, iterations=1000)
    node_colors = n1*[first_community_color] + n2*[second_community_color] + \
        n3*[third_community_color] + n4*[fourth_community_color]

    node_subsets = [range(0, sizes[0]),
                    range(sizes[0], sizes[0] + sizes[1]),
                    range(sizes[1], sizes[1] + sizes[2]),
                    range(sizes[2], sizes[2] + sizes[3])]

    # Get layout
    pos, edge_bunchs = draw.clustered_layout(G_SBM, node_subsets,
                                             centroids, radius_scale)
    # draw.draw_networks(G_SBM, pos, ax_SBM,
    #                    mu=mu,
    #                    edge_color=edge_color,
    #                    edge_width=edge_width,
    #                    edge_alpha=edge_alpha,
    #                    use_edge_weigth=use_edge_weigth,
    #                    node_width=node_width,
    #                    node_size=node_size,
    #                    node_border_color=node_border_color,
    #                    node_color=node_colors,
    #                    node_alpha=node_alpha,
    #                    arrow_scale=arrow_scale,
    #                    loop_radius=loop_radius)

    for i, nodes in enumerate(node_subsets):
        nx.draw_networkx_nodes(G_SBM, pos=pos, node_size=node_size,
                               nodelist=nodes, node_color=node_colors[i])

    colors = ['#bdbdbd', '#bdbdbd', '#bdbdbd', '#bdbdbd', '#bdbdbd', '#bdbdbd',
              '#bdbdbd', '#bdbdbd','#bdbdbd', '#bdbdbd', '#bdbdbd', '#bdbdbd',
              '#bdbdbd', '#bdbdbd', '#bdbdbd', '#bdbdbd','#bdbdbd', '#bdbdbd',
              '#bdbdbd', '#bdbdbd']
    for i, edges in enumerate(edge_bunchs):
        nx.draw_networkx_edges(G_SBM, pos=pos, alpha=0.2, edgelist=edges,
                               edge_color=colors[i])
    plt.axis("off")

    plt.show()


def draw_erdos_article_synchro(N, p):
    # Draw Erdos-Renyi
    ax_erdos = plt.subplot(111)
    G_erdos = nx.erdos_renyi_graph(N, p)
    pos = nx.spring_layout(G_erdos)
    draw.draw_networks(G_erdos, pos, ax_erdos,
                       mu=mu,
                       edge_color=edge_color,
                       edge_width=edge_width,
                       edge_alpha=edge_alpha,
                       use_edge_weigth=use_edge_weigth,
                       node_width=node_width,
                       node_size=node_size,
                       node_border_color=node_border_color,
                       node_color=first_community_color,
                       node_alpha=node_alpha,
                       arrow_scale=arrow_scale,
                       loop_radius=loop_radius)
    plt.show()


def draw_reduced_erdos_article_synchro():
    # Draw reduced Erdos
    ax_red_erdos = plt.subplot(242)
    G_red_erdos = nx.from_numpy_matrix(np.array([[1]]))
    pos = nx.spring_layout(G_red_erdos)
    draw.draw_networks(G_red_erdos, pos, ax_red_erdos,
                       mu=mu,
                       edge_color=edge_color,
                       edge_width=edge_width,
                       edge_alpha=edge_alpha,
                       use_edge_weigth=use_edge_weigth,
                       node_width=node_width,
                       node_size=reduced_node_size,
                       node_border_color=node_border_color,
                       node_color=first_community_color,
                       node_alpha=node_alpha,
                       arrow_scale=arrow_scale,
                       loop_radius=loop_radius)
    plt.show()


def draw_bipartite_article_synchro(sizes, p):
    # Draw SBM
    ax_bipartite = plt.subplot(111)
    n1, n2 = sizes
    pq = [[0.0, p],
          [p, 0.0]]
    G_bipartite = nx.stochastic_block_model(sizes, pq)
    pos = nx.drawing.bipartite_layout(G_bipartite,
                                      list(G_bipartite.nodes())[0:n1],
                                      align="horizontal")
    node_colors = n1 * [first_community_color] + n2 * [second_community_color]
    draw.draw_networks(G_bipartite, pos, ax_bipartite,
                       mu=mu,
                       edge_color=edge_color,
                       edge_width=edge_width,
                       edge_alpha=edge_alpha,
                       use_edge_weigth=use_edge_weigth,
                       node_width=node_width,
                       node_size=node_size,
                       node_border_color=node_border_color,
                       node_color=node_colors,
                       node_alpha=node_alpha,
                       arrow_scale=arrow_scale,
                       loop_radius=loop_radius)
    plt.show()


def draw_reduced_bipartite_article_synchro():

    # Draw reduced SBM
    ax_red_bipartite = plt.subplot(111)
    G_red_bipartite = nx.from_numpy_matrix(np.array([[0, 1], [1, 0]]))
    pos = nx.spring_layout(G_red_bipartite)
    node_colors = [first_community_color, second_community_color]
    draw.draw_networks(G_red_bipartite, pos, ax_red_bipartite,
                       mu=mu,
                       edge_color=edge_color,
                       edge_width=edge_width,
                       edge_alpha=edge_alpha,
                       use_edge_weigth=use_edge_weigth,
                       node_width=node_width,
                       node_size=node_size,
                       node_border_color=node_border_color,
                       node_color=node_colors,
                       node_alpha=node_alpha,
                       arrow_scale=arrow_scale,
                       loop_radius=loop_radius)
    plt.show()


def draw_sbm_article_synchro(sizes, pq, centroids, radius_scale):
    # Draw SBM

    ax_SBM = plt.subplot(111)
    n1, n2 = sizes
    G_SBM = nx.stochastic_block_model(sizes, pq)
    # pos = nx.layout.spectral_layout(G_SBM)
    # pos = nx.spring_layout(G_SBM, pos=pos, iterations=1000)
    node_colors = n1 * [first_community_color] + n2 * [second_community_color]
    node_subsets = [range(0, sizes[0]), range(sizes[0], sizes[0] + sizes[1])]

    # Get layout
    pos, edge_bunchs = draw.clustered_layout(G_SBM, node_subsets,
                                             centroids, radius_scale)

    # Draw
    # plt.figure(figsize=(2, 2))

    # node_colors = [first_community_color, second_community_color]
    # for i, nodes in enumerate(node_subsets):
    #     nx.draw_networkx_nodes(G_SBM, pos=pos, node_size=node_size,
    #                            nodelist=nodes, node_color=node_colors[i])
    #
    # colors = ['#bdbdbd', '#bdbdbd', '#bdbdbd', '#bdbdbd']
    # for i, edges in enumerate(edge_bunchs):
    #     nx.draw_networkx_edges(G_SBM, pos=pos, alpha=0.2, edgelist=edges,
    #                            edge_color=colors[i])
    #
    # plt.axis("off")
    # plt.show()

    draw.draw_networks(G_SBM, pos, ax_SBM,
                       mu=mu,
                       edge_color=edge_color,
                       edge_width=edge_width,
                       edge_alpha=edge_alpha,
                       use_edge_weigth=use_edge_weigth,
                       node_width=node_width,
                       node_size=node_size,
                       node_border_color=node_border_color,
                       node_color=node_colors,
                       node_alpha=node_alpha,
                       arrow_scale=arrow_scale,
                       loop_radius=loop_radius)
    plt.show()


def draw_reduced_sbm_synchro():
    # Draw reduced SBM
    ax_red_SBM = plt.subplot(111)
    G_red_SBM = nx.from_numpy_matrix(np.array([[0, 1], [1, 0]]))
    pos = nx.spring_layout(G_red_SBM)
    node_colors = [first_community_color, second_community_color]
    draw.draw_networks(G_red_SBM, pos, ax_red_SBM,
                       mu=mu,
                       edge_color=edge_color,
                       edge_width=edge_width,
                       edge_alpha=edge_alpha,
                       use_edge_weigth=use_edge_weigth,
                       node_width=node_width,
                       node_size=node_size,
                       node_border_color=node_border_color,
                       node_color=node_colors,
                       node_alpha=node_alpha,
                       arrow_scale=arrow_scale,
                       loop_radius=loop_radius)
    plt.show()


def draw_two_star_article_synchro(sizes, pq):
    # Draw two-star
    ax_ts = plt.subplot(111)
    N = sum(sizes)
    n1, n2 = sizes
    A = np.zeros((N, N))
    ii = 0
    for i in range(0, len(sizes)):
        jj = 0
        for j in range(0, len(sizes)):
            if i == j:
                A[ii:ii + sizes[i], jj:jj + sizes[j]] \
                    = pq[i][j]*nx.to_numpy_matrix(nx.star_graph(
                                                  sizes[i]-1,
                                                  create_using=None))
            else:
                A[ii, jj] = pq[i][j]
            jj += sizes[j]
        ii += sizes[i]

    G_ts = nx.from_numpy_matrix(A)
    pos = nx.layout.kamada_kawai_layout(G_ts)
    # pos = nx.layout.spectral_layout(G_ts, pos=pos)
    pos = nx.spring_layout(G_ts, pos=pos, iterations=1000)
    node_colors = ["#deebf7"] + (n1-1) * [first_community_color] \
        + ["#fee6ce"] + (n2-1) * [second_community_color]
    draw.draw_networks(G_ts, pos, ax_ts,
                       mu=mu,
                       edge_color=edge_color,
                       edge_width=edge_width,
                       edge_alpha=edge_alpha,
                       use_edge_weigth=use_edge_weigth,
                       node_width=node_width,
                       node_size=node_size,
                       node_border_color=node_border_color,
                       node_color=node_colors,
                       node_alpha=node_alpha,
                       arrow_scale=arrow_scale,
                       loop_radius=loop_radius)
    plt.show()


def draw_reduced_two_star_article_synchro():
    # Draw reduced two-star
    ax_red_ts = plt.subplot(111)
    G_red_ts = nx.from_numpy_matrix(np.array([[0, 1, 0, 0],
                                              [1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0]]))
    pos = nx.spring_layout(G_red_ts)
    node_colors = [first_community_color, first_community_color,
                   second_community_color, second_community_color]
    draw.draw_networks(G_red_ts, pos, ax_red_ts,
                       mu=mu,
                       edge_color=edge_color,
                       edge_width=edge_width,
                       edge_alpha=edge_alpha,
                       use_edge_weigth=use_edge_weigth,
                       node_width=node_width,
                       node_size=node_size,
                       node_border_color=node_border_color,
                       node_color=node_colors,
                       node_alpha=node_alpha,
                       arrow_scale=arrow_scale,
                       loop_radius=loop_radius)

    plt.show()


if __name__ == '__main__':
    plt.figure(figsize=(1, 2))
    # draw_erdos_article_synchro(25, 0.4)
    # draw_bipartite_article_synchro([10, 8], 0.2)
    # draw_sbm_article_synchro([60, 40], [[0.6, 0.01], [0.01, 0.6]],
    #                          [(0, 15), (0, 0)], [0.8, 0.75])
    # draw_sbm_intro_article_synchro([20, 10, 8, 10], [[0.3, 0.1, 0,   0],
    #                                                  [0.1, 0.3, 0.2, 0.1],
    #                                                  [0,   0.2, 0.3, 0],
    #                                                  [0,   0.1, 0,   0.3]],
    #                                [(-10, 0), (0, 15), (15, 15), (-15, 15)],
    #                                [1, 1, 1, 1])
    # draw_two_star_article_synchro([10, 8], [[1, 1],[1, 1]])
