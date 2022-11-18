from dynamics.get_reduction_errors import *
from plots.plot_complete_vs_reduced import *
import matplotlib.pyplot as plt
import json
import numpy as np
from tkinter import messagebox
import itertools

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
# labelpad = 10
# ylim = [0, 1.05]
# xlim = [-0.05, 8.05]
# xticks = [0, 4, 8]
# yticks = [0, 0.5, 1.0]
# x, y = 0.05, 0.02

labelpad = 15
ylim = [0, 1.05]
xlim = [-0.2, 4.1]
xticks = [0, 2, 4]
yticks = [0, 0.5, 1.0]
x, y = 0.05, 0.02

# labelpad = 30
# ylim = [0, 1.05]
# xlim = [0.95, 4.05]
# xticks = [1, 2, 3, 4]
# yticks = [0, 0.5, 1.0]
# x, y = 2.2, 0.02

# labelpad = 30
# ylim = [0.2, 1.05]
# xlim = [1.95, 5.05]
# yticks = [0.4, 0.6, 0.8, 1.0]
# xticks = [2, 3, 4, 5]
# x, y = 2.05, 0.22
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

""" Two-triangle 2D """
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_21_03h35min24sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_two_triangles_2D' \
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_29_20h08min36sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_two_triangle_2D_V_K_all_nodes'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_29_21h46min11sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_one_less_node'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_29_22h45min11sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_two_triangles_2D_perturbed'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_02_01_04h15min03sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_snmf_and_onmf'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_02_04_21h48min34sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_multiple_inits_V_K_perturbed'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_02_05_01h20min40sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_multiple_inits_V_K_one_less_node'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_02_06_03h20min32sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_snmf_onmf_multiple_inits'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_02_06_03h48min26sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_not_aligned'
# file = '2020_02_07_11h42min22sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_more_inits_onmf_snmf'
# file = '2020_09_30_18h48min28sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_2D_two_triangles'
# file = "data/kuramoto/kuramoto_secIIID_article/" \
#        "2020_10_02_23h30min05sec_multiple_synchro_transition" \
#        "_dictionary_kuramoto_2D_two_triangles"
# file = "data/kuramoto/kuramoto_secIIID_article/" \
#        "2020_10_05_05h15min21sec_multiple_synchro_transition_dictionary" \
#        "_kuramoto_2D_two_triangles"
file = "data/kuramoto/kuramoto_secIIID_article/" \
       "2020_10_09_20h16min05sec_multiple_synchro_" \
       "transition_dictionary_kuramoto_2D_two_triangles"

""" Two-triangle 3D """
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_21_07h03min26sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_two_triangles_3D'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_29_22h43min46sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_two_triangles_3D_V0_V1_V3' \
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_02_01_04h24min01sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_snmf_and_onmf'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_02_06_03h38min49sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_snmf_onmf_multiple_inits'
# file = "data/kuramoto/kuramoto_secIIID_article/" \
#        "2020_02_07_19h26min47sec_multiple_synchro_transition" \
#        "_dictionary_kuramoto_multiple_inits_V_K_one_less_node"
# file = "data/kuramoto/kuramoto_secIIID_article/" \
#        "2020_10_04_19h00min02sec_multiple_synchro_transition_dictionary" \
#        "_kuramoto_3D_two_triangles"
# file = "data/kuramoto/kuramoto_secIIID_article/" \
#        "2020_10_04_23h47min16sec_multiple_synchro_transition_dictionary" \
#        "_kuramoto_3D_two_triangles"
# file = "data/kuramoto/kuramoto_secIIID_article/" \
#        "2020_10_05_04h25min06sec_multiple_synchro_transition_dictionary" \
#        "_kuramoto_3D_two_triangles"

""" Small bipartite 2D """
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_21_23h18min55sec_multiple_synchro_transition_' \
#        'dictionary_kuramoto_small_bipartite_2D'
#
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_22_02h56min17sec_multiple_synchro_transition' \
#        '_dictionary_kuramoto_small_bipartite_2D'
#
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_23_19h17min18sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_small_bipartite_deg_freq_2D'
#
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_23_19h59min24sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_small_bipartite_other_deg_freq_2D'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_30_01h05min40sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_small_bipartite_2D'
# file = 'data/kuramoto/kuramoto_secIIID_article/2020_02_01_01h42min33sec' \
#        '_multiple_synchro_transition_dictionary_kuramoto_snmf_and_onmf'

""" Small bipartite 3D """
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_21_19h39min58sec_multiple_synchro' \
#        '_transition_dictionary_kuramoto_small_bipartite_3D'
#
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_22_02h30min29sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_small_bipartite_3D'
#
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_22_02h37min18sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_small_bipartite_3D_dom_veps'
#
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_23_19h38min41sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_small_bipartite_deg_freq_3D'
#
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_23_20h13min29sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_small_bipartite_other_deg_freq_3D'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_01_30_01h19min23sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_small_bipartite_3D'
# file = 'data/kuramoto/kuramoto_secIIID_article/' \
#        '2020_02_01_01h59min22sec_multiple_synchro_transition_dictionary' \
#        '_kuramoto_snmf_and_onmf'

""" Bipartite, n=2"""
# file = "2020_02_09_18h04min11sec_multiple_synchro_transition" \
#        "_dictionary_kuramoto_first_test_bipartite_pout_0_2"
# file = "2020_02_10_05h55min18sec_multiple_synchro_transition" \
#        "_dictionary_kuramoto_better_sigma_array_bipartite_pout_0_2"
# file = "2020_02_10_20h56min18sec_multiple_synchro_transition" \
#        "_dictionary_kuramoto_bipartite_2D_one_random_init_cond"

""" SBM, n = 2 """
# file = "2020_02_10_06h20min54sec_multiple_synchro_transition" \
#        "_dictionary_kuramoto_first_test_SBM_pin_0_8_pout_0_2"

with open(file + ".json") \
        as json_data:
    multiple_synchro_transition_dictionary = json.load(json_data)

sigma_array = np.array(multiple_synchro_transition_dictionary["sigma_array"])
# targets_possibilities = ["W", "K", "A", "WK", "KW", "AW", "WA", "KA",
#                          "AK", "WKA", "KWA", "AWK", "WAK", "KAW", "AKW"]
targets_possibilities = ["W", "WK", "WA", "WKA", "WAK",
                         "K", "KW", "KA", "KWA", "KAW",
                         "A", "AW", "AK", "AWK", "AKW"]

colors = [reduced_first_community_color, reduced_second_community_color,
          reduced_third_community_color]

plt.figure(figsize=(8, 4.5))

c1 = "#9ecae1"
c2 = "#fdd0a2"
c3 = "#a1d99b"
marker = itertools.cycle(('*', '^', 'P', 'o', 's'))
markersize = itertools.cycle((140, 150, 150, 200, 200))
color_markers = 5 * [c1] + 5 * [c2] + 5 * [c3]

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

    ax = plt.subplot(3, 5, i+1)
    RMSE_transition = RMSE(np.array(r), np.array(R))
    plot_multiple_transitions_complete_vs_reduced(
        ax, sigma_array, r, R, total_color,
        color, linewidth)
    ax.text(x=x, y=y, s=targets_choices + "\n" + "RMSE $\\approx$ {}"
            .format("%.3f" % np.round(RMSE_transition, 3)),
            fontsize=fontsize-1, zorder=15)
    # plt.scatter(xticks[1]+1.5, 0.6,
    #             s=next(markersize),
    #             color=color_markers[i],
    #             marker=next(marker),
    #             edgecolors='#525252')
    # "$T_1 = {}$ \n $T_2 = {}$\n $T_3 = {}$"
    left_bottom_axis_settings(ax, xlabel, ylabel, xlim, ylim,
                              (0.5, -0.3), (-0.5, 0.45),
                              fontsize, labelsize, linewidth)
    # left_bottom_axis_settings(ax, xlabel, ylabel, xlim, ylim,
    #                           (0.5, -0.35), (-0.3, 0.45),
    #                           fontsize, labelsize, linewidth)
    plt.xticks(xticks)
    plt.yticks(yticks)

plt.tight_layout()
fig = plt.gcf()
plt.show()
if messagebox.askyesno("Python", "Would you like to save the plot ?"):
    fig.savefig(file + "_plot.png")
    fig.savefig(file + "_plot.pdf")
