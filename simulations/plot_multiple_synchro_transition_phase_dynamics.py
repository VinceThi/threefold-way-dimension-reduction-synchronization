from synch_predictions.dynamics.get_reduction_errors import *
from synch_predictions.plots.plot_complete_vs_reduced import *
from synch_predictions.graphs.get_reduction_matrix_and_characteristics import *
import matplotlib.pyplot as plt
import json
import numpy as np
# from tkinter import messagebox
# import tkinter.simpledialog
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools


win_kur_theta = 0

if win_kur_theta:

    # file1 = "2020_02_14_11h39min14sec_multiple_synchro_transition" \
    #         "_dictionary_winfree_on_bipartite_2D"
    # file2 = "2020_02_14_12h08min49sec_multiple_synchro_transition" \
    #         "_dictionary_winfree_on_SBM_2D"
    # file3 = "2020_02_14_11h43min38sec_multiple_synchro_transition" \
    #         "_dictionary_kuramoto_on_bipartite_2D"
    # file4 = "2020_02_14_12h14min32sec_multiple_synchro_transition" \
    #         "_dictionary_kuramoto_on_SBM_2D"
    # file5 = "2020_02_14_12h46min08sec_multiple_synchro_transition" \
    #         "_dictionary_theta_on_bipartite_2D"
    # file6 = "2020_02_14_12h45min57sec_multiple_synchro_transition" \
    #         "_dictionary_theta_on_SBM_2D"
    #
    # files = [file1, file3, file5, file2, file4, file6]

    setup_str_list = ["bipartite_winfree", "bipartite_kuramoto",
                      "bipartite_theta", "SBM_winfree",
                      "SBM_kuramoto", "SBM_theta"]

    bbox_to_anchors1 = [(.5, .55, .5, .5), (.5, .55, .5, .5),
                        (-0.02, .55, .5, .5), (0.05, .15, .5, .5),
                        (0, .15, .5, .5), (-0.05, .5, .5, .5)]
    bbox_to_anchors2 = [(.5, .1, .5, .5), (.5, .1, .5, .5),
                        (-0.02, .1, .5, .5), (.5, .15, .5, .5),
                        (.5, .15, .5, .5), (.5, .5, .5, .5)]
    ylim0 = [[0, 1.02], [0, 1.02], [0.5, 1.02],
             [0, 1.02], [0, 1.02], [0, 1.02]]
    ylim1 = [[0, 1.1], [0, 1.1], [0.4, 1.1],
             [0, 1.1], [0, 1.1], [0, 1.1]]
    ylim2 = [[0, 1.1], [0, 1.1], [0.4, 1.1],
             [0, 1.1], [0, 1.1], [0, 1.1]]
    yticks = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    yticks0 = [[0, 0.5, 1], [0, 0.5, 1], [0.5, 0.75, 1],
               [0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1]]

    dynamics = ["Winfree", "Kuramoto", "theta"]
    graph_label = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

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
    alpha = 0.3
    marker = "o"
    # y_lim = [0, 1.02]
    nb_instances = 1
    # reduced_first_community_color = "#9ecae1"
    # reduced_second_community_color = "#fdd0a2"
    # reduced_third_community_color = "#a1d99b"
    # total_color = "#525252"
    # fontsize = 12
    # inset_fontsize = 9
    # fontsize_legend = 12
    # labelsize = 12
    # inset_labelsize = 9
    # linewidth = 2
    # # labelpad = 10
    # # ylim = [0, 1.05]
    # # xlim = [-0.05, 8.05]
    # # xticks = [0, 4, 8]
    # # yticks = [0, 0.5, 1.0]
    # # x, y = 0.05, 0.02
    #
    # # labelpad = 10
    # # ylim = [0, 1.05]
    # # xlim = [-0.05, 4.05]
    # # xticks = [0, 2, 4]
    # # yticks = [0, 0.5, 1.0]
    # # x, y = 0.05, 0.02
    #
    # labelpad = 30
    # ylim = [0, 1.05]
    # xlim = [0.95, 4.05]
    # xticks = [1, 2, 3, 4]
    # yticks = [0, 0.5, 1.0]
    # x, y = 2.2, 0.02
    #
    # # labelpad = 30
    # # ylim = [0.2, 1.05]
    # # xlim = [1.95, 5.05]
    # # yticks = [0.4, 0.6, 0.8, 1.0]
    # # xticks = [2, 3, 4, 5]
    # # x, y = 2.05, 0.22
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(8, 5))

    for i, setup_string in enumerate(setup_str_list):
        # with open("data/z_synchro_transitions_2020_one_instance/" +
        #           file + ".json") \
        #         as json_data:
        #     synchro_transition_dictionary = json.load(json_data)
        file = get_transitions_realizations_dictionary_absolute_path(
                       setup_string)
        with open("data/synchro_transitions_multiple_realizations/" +
                  file + ".json") \
                as json_data:
            synchro_transitions_dictionary = json.load(json_data)

        infos_realizations_dictionary_path = \
            get_infos_realizations_dictionary_absolute_path(
                setup_string)
        with open(f'C:/Users/thivi/Documents/GitHub/network-synch/'
                  f'synch_predictions/simulations/data/'
                  f'synchro_transitions_multiple_realizations/'
                  f'{infos_realizations_dictionary_path}.json') as json_data:
            infos_realizations_dictionary = json.load(json_data)

        if i > 2:
            xlabel = "$\\sigma$"
        else:
            xlabel = ""

        if i == 0 or i == 3:
            ylabel = "$\\langle R\,\\rangle$"
        else:
            ylabel = ""

        number_realizations = \
            infos_realizations_dictionary["number_realizations"]
        sigma_array = np.array(synchro_transitions_dictionary["sigma_array"])

        ax = plt.subplot(2, 3, i+1)
        ax.title.set_position((0.5, 1.05))

        r_matrix = np.array(synchro_transitions_dictionary["r_list"])
        R_matrix = np.array(synchro_transitions_dictionary["R_list"])
        # if setup_string == "bipartite_theta" or "bipartite_SBM":
        #     print(R_matrix)
        r1_matrix = np.zeros(np.shape(r_matrix))
        r2_matrix = np.zeros(np.shape(r_matrix))
        R1_matrix = np.zeros(np.shape(r_matrix))
        R2_matrix = np.zeros(np.shape(r_matrix))
        for j in range(number_realizations):
            r1_matrix[j] = \
                synchro_transitions_dictionary["r_mu_matrix_list"][j][0]
            r2_matrix[j] = \
                synchro_transitions_dictionary["r_mu_matrix_list"][j][1]
            R1_matrix[j] = \
                synchro_transitions_dictionary["R_mu_matrix_list"][j][0]
            R2_matrix[j] = \
                synchro_transitions_dictionary["R_mu_matrix_list"][j][1]

        if i < 3:
            ax.title.set_text(f"{dynamics[i]}\n   {graph_label[i]}")
        else:
            ax.title.set_text(f"{graph_label[i]}")

        if setup_string == "bipartite_theta":
            plt.xticks([0, 5, 10])
            xlim = [-0.3, 10]

        elif setup_string == "SBM_theta":
            r_matrix = r_matrix[:, :-7]
            R_matrix = R_matrix[:, :-7]
            r1_matrix = r1_matrix[:, :-7]
            r2_matrix = r2_matrix[:, :-7]
            R1_matrix = R1_matrix[:, :-7]
            R2_matrix = R2_matrix[:, :-7]
            sigma_array = sigma_array[:-7]
            plt.xticks([0, 5, 10])
            xlim = [-0.3, 10]
        else:
            if setup_string == "bipartite_kuramoto":
                plt.xticks([0, 5, 10, sigma_array[-1]])
            elif setup_string == "SBM_winfree":
                plt.xticks([0, 1, 2, 3, 4, 5])
            else:
                plt.xticks([0, 5, sigma_array[-1]])
            xlim = [-0.3, sigma_array[-1]]
        plot_transitions_complete_vs_reduced(ax, sigma_array,
                                             r_matrix, R_matrix,
                                             "#252525", "#969696",
                                             alpha, marker, s, linewidth,
                                             number_realizations)

        left_bottom_axis_settings(ax, xlabel, ylabel,
                                  xlim, ylim0[i],
                                  (0.5, -0.15), (-0.3, 0.45), fontsize,
                                  labelsize, linewidth)
        plt.yticks(yticks0[i])

        axins1 = inset_axes(ax, width="50%", height="50%",
                            bbox_to_anchor=bbox_to_anchors1[i],
                            bbox_transform=ax.transAxes, loc=4)
        plot_transitions_complete_vs_reduced(axins1, sigma_array,
                                             r1_matrix, R1_matrix,
                                             first_community_color,
                                             reduced_first_community_color,
                                             alpha, marker, s-5, linewidth,
                                             number_realizations)
        left_bottom_axis_settings(axins1, "$\\sigma$",
                                  "$\\langle R_1 \\rangle$",
                                  xlim, ylim1[i],
                                  (0.5, -0.2), (-0.35, 0.25),
                                  inset_fontsize, inset_labelsize,
                                  linewidth)
        if setup_string == "bipartite_theta":
            plt.xticks([0, 10])
        elif setup_string == "SBM_theta": 
            plt.xticks([0, 10])           
        else:
            plt.xticks([0, sigma_array[-1]])

        axins2 = inset_axes(ax, width="50%", height="50%",
                            bbox_to_anchor=bbox_to_anchors2[i],
                            bbox_transform=ax.transAxes, loc=4)
        plot_transitions_complete_vs_reduced(axins2, sigma_array,
                                             r2_matrix, R2_matrix,
                                             second_community_color,
                                             reduced_second_community_color,
                                             alpha, marker, s-5, linewidth,
                                             number_realizations)
        left_bottom_axis_settings(axins2, "$\\sigma$",
                                  "$\\langle R_2 \\rangle$",
                                  xlim, ylim2[i],
                                  (0.5, -0.2), (-0.35, 0.25),
                                  inset_fontsize, inset_labelsize,
                                  linewidth)
        if setup_string == "bipartite_theta":
            plt.xticks([0, 10])
        elif setup_string == "SBM_theta":
            plt.xticks([0, 10])
        else:
            plt.xticks([0, sigma_array[-1]])

        plt.tight_layout()
    plt.show()

else:

    # file = "2020_02_10_06h20min54sec_multiple_synchro_transition" \
    #        "_dictionary_kuramoto_first_test_SBM_pin_0_8_pout_0_2"
    # file = "2020_02_28_15h16min34sec_multiple_synchro_transition" \
    #        "_dictionary_kuramoto_2D_two_triangles_article"

    # file = "2020_02_21_19h04min15sec_multiple_synchro_transition" \
    #        "_dictionary_kuramoto_3D_two_triangles_article"
    # file = "2020_02_21_23h42min39sec_multiple_synchro_transition" \
    #        "_dictionary_kuramoto_3D_two_triangles_article"
    # file = "2020_02_28_16h50min22sec_multiple_synchro_transition" \
    #        "_dictionary_kuramoto_3D_two_triangles_article"
    # file = "2020_02_28_21h07min59sec_multiple_synchro_transition" \
    #        "_dictionary_kuramoto_3D_two_triangles_article"

    """ In the article """
    file = "2020_02_21_03h14min07sec_multiple_synchro_transition" \
           "_dictionary_kuramoto_2D_two_triangles_article"
    # file = "2020_02_29_17h42min33sec_multiple_synchro_transition" \
    #        "_dictionary_kuramoto_3D_two_triangles"

    reduced_first_community_color = "#9ecae1"
    reduced_second_community_color = "#fdd0a2"
    reduced_third_community_color = "#a1d99b"
    total_color = "#525252"
    fontsize = 12
    inset_fontsize = 9
    fontsize_legend = 12
    labelsize = 12
    inset_labelsize = 9
    linewidth = 2
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # labelpad = 10
    # ylim = [0, 1.05]
    # xlim = [-0.05, 8.05]
    # xticks = [0, 4, 8]
    # yticks = [0, 0.5, 1.0]
    # x, y = 0.05, 0.02

    # labelpad = 10
    # ylim = [0, 1.05]
    # xlim = [-0.05, 4.05]
    # xticks = [0, 2, 4]
    # yticks = [0, 0.5, 1.0]
    # x, y = 0.05, 0.02
    plot_markers = False
    ylim = [0, 1.05]
    xlim = [-0.2, 4.05]
    xticks = [0, 2, 4]      
    yticks = [0, 0.5, 1.0]  
    x, y = 0.2, 0.03

    with open("data/kuramoto/kuramoto_secIIID_article/" + file + ".json") \
            as json_data:
        multiple_synchro_transition_dictionary = json.load(json_data)

    targets_possibilities = ["W", "WK", "WA", "WKA", "WAK",
                             "K", "KW", "KA", "KWA", "KAW",
                             "A", "AW", "AK", "AWK", "AKW"]

    # targets_possibilities = ["W", "A"]

    colors = [reduced_first_community_color, reduced_second_community_color,
              reduced_third_community_color]

    sigma_array = np.array(
        multiple_synchro_transition_dictionary["sigma_array"])

    plt.figure(figsize=(9, 4.5))
    # plt.figure(figsize=(5, 3))

    c1 = "#9ecae1"
    c2 = "#fdd0a2"
    c3 = "#a1d99b"
    marker = itertools.cycle(('*', '^', 'P', 'o', 's'))
    markersize = itertools.cycle((140, 150, 150, 200, 200))
    color_markers = 5 * [c1] + 5 * [c2] + 5 * [c3]
    # color_markers = [c1, c3]
    for i, targets_string in enumerate(targets_possibilities):
        r = multiple_synchro_transition_dictionary[f"r_{targets_string}"]
        R = multiple_synchro_transition_dictionary[f"R_{targets_string}"]
        xlabel = ""
        if i < 5:
            color = colors[0]
        elif 5 <= i < 10:
            color = colors[1]
        else:
            color = colors[2]
            xlabel = r"$\sigma$"

        # if not i:
        #     color = colors[0]
        #     xlabel = r"$\sigma$"
        #
        # else:
        #     color = colors[2]
        #     xlabel = r"$\sigma$"
                       
        if not i % 5:
            ylabel = r"$\langle R \rangle_t$"
        else:
            ylabel = ""

        if len(targets_string) == 1:
            T_1 = targets_string
            targets_choices = f"${T_1}$"
        elif len(targets_string) == 2:
            T_1, T_2 = list(targets_string)
            targets_choices = f"${T_1} \\rightarrow {T_2}$"
        else:
            T_1, T_2, T_3 = list(targets_string)
            targets_choices = f"${T_1} \\rightarrow {T_2} \\rightarrow {T_3}$"

        ax = plt.subplot(3, 5, i + 1)
        # ax = plt.subplot(1, 2, i + 1)
        RMSE_transition = RMSE(np.array(r), np.array(R))
        plot_multiple_transitions_complete_vs_reduced(
            ax, sigma_array, r, R, total_color,
            color, linewidth)
        ax.text(x=x, y=y, s=targets_choices + "\n" + "RMSE $\\approx$ {}"
                .format("%.3f" % np.round(RMSE_transition, 3)),
                fontsize=fontsize-1, zorder=15)
        if plot_markers:
            plt.scatter(xticks[1] + 1.5, 0.6,
                        s=next(markersize),
                        color=color_markers[i],
                        marker=next(marker),
                        edgecolors='#525252')
        # "$T_1 = {}$ \n $T_2 = {}$\n $T_3 = {}$"
        # left_bottom_axis_settings(ax, xlabel, ylabel, xlim, ylim,
        #                           (0.5, -0.25), (-0.3, 0.45),
        #                           fontsize, labelsize, linewidth)
        left_bottom_axis_settings(ax, xlabel, ylabel, xlim, ylim,
                                  (0.5, -0.35), (-0.4, 0.45),
                                  fontsize, labelsize, linewidth)
        plt.xticks(xticks)
        plt.yticks(yticks)

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()


# if messagebox.askyesno("Python",
#                        "Would you like to save the plot ?"):
#     window = tkinter.Tk()
#     window.withdraw()  # hides the window
#     file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
#     timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
#     fig.savefig("data/kuramoto/"
#                 "{}_{}_complete_vs_reduced_kuramoto_2D"
#                 ".png".format(timestr, file))
#     fig.savefig("data/kuramoto/"
#                 "{}_{}_complete_vs_reduced_kuramoto_2D"
#                 ".pdf".format(timestr, file))
