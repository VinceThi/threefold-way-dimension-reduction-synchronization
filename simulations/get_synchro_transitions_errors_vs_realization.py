from graphs.get_reduction_matrix_and_characteristics import *
import json
from graphs.special_graphs import mean_SBM
import matplotlib.pyplot as plt

setup_str_list = ["bipartite_winfree", "bipartite_kuramoto",
                  "bipartite_theta", "SBM_winfree",
                  "SBM_kuramoto", "SBM_theta"]

frobenius_norm_matrix = np.zeros((len(setup_str_list), 50))
rmse_transition_matrix = np.zeros((len(setup_str_list), 50))

for i, setup in enumerate(setup_str_list):

    print(f"{setup}")

    """ Import data """
    realizations_dictionary_str = f"{setup}"
    realizations_dictionary_path = \
        get_realizations_dictionary_absolute_path(realizations_dictionary_str)
    with open(f'C:/Users/thivi/Documents/GitHub/network-synch/'
              f'synch_predictions/simulations/data/'
              f'synchro_transitions_multiple_realizations/'
              f'{realizations_dictionary_path}.json') as json_data:
        realizations_dictionary = json.load(json_data)

    infos_realizations_dictionary_str = f"{setup}"
    infos_realizations_dictionary_path = \
        get_infos_realizations_dictionary_absolute_path(
            infos_realizations_dictionary_str)
    with open(f'C:/Users/thivi/Documents/GitHub/network-synch/'
              f'synch_predictions/simulations/data/'
              f'synchro_transitions_multiple_realizations/'
              f'{infos_realizations_dictionary_path}.json') as json_data:
        infos_realizations_dictionary = json.load(json_data)

    file = get_transitions_realizations_dictionary_absolute_path(
                           f"{setup}")
    with open("data/synchro_transitions_multiple_realizations/" +
              file + ".json") \
            as json_data:
        synchro_transitions_dictionary = json.load(json_data)

    r_realizations = synchro_transitions_dictionary["r_list"]
    R_realizations = synchro_transitions_dictionary["R_list"]
    adjacency_matrix_realizations = \
        realizations_dictionary["adjacency_matrix_realizations"]
    sizes = infos_realizations_dictionary["sizes"]
    pq = infos_realizations_dictionary["affinity_matrix"]
    mean_A = mean_SBM(sizes, pq)
    dmax = np.zeros(np.shape(mean_A))
    if pq[0][0] != 0:
        p_mat_up = max([pq[0][0], 1-pq[0][0]])*np.ones((sizes[0], sizes[0]))
        p_mat_low = max([pq[1][1], 1-pq[1][1]])*np.ones((sizes[1], sizes[1]))
        q_mat_up = max([pq[0][1], 1-pq[0][1]])*np.ones((sizes[0], sizes[1]))
        q_mat_low = max([pq[1][0], 1-pq[1][0]])*np.ones((sizes[1], sizes[0]))
    else:
        p_mat_up = np.zeros((sizes[0], sizes[0]))
        p_mat_low = np.zeros((sizes[1], sizes[1]))
        q_mat_up = max([pq[0][1], 1-pq[0][1]])*np.ones((sizes[0], sizes[1]))
        q_mat_low = max([pq[1][0], 1-pq[1][0]])*np.ones((sizes[1], sizes[0]))

    A_max = np.block([[p_mat_up, q_mat_up], [q_mat_low, p_mat_low]])
    norm_A_max = np.linalg.norm(A_max)
    print(norm_A_max)
    # mean_k = np.sum(mean_A)
    # print(mean_k)
    # print(mean_A)

    frobenius_norm_list = []
    rmse_transition_list = []
    for j, A in enumerate(adjacency_matrix_realizations):
        r = np.nan_to_num(np.array(r_realizations[j]))
        R = np.nan_to_num(np.array(R_realizations[j]))
        frobenius_norm_list.append(np.linalg.norm(A - mean_A)/norm_A_max)
        rmse_transition_list.append(rmse(r, R))

    frobenius_norm_matrix[i, :] = np.array(frobenius_norm_list)
    rmse_transition_matrix[i, :] = np.array(rmse_transition_list)

# print(rmse_transition_matrix)
first_community_color = "#2171b5"
reduced_first_community_color = "#9ecae1"
second_community_color = "#f16913"
reduced_second_community_color = "#fdd0a2"
third_community_color = "#31a354"
fourth_community_color = "#756bb1"
colors = [first_community_color, second_community_color, third_community_color]
fontsize = 12
inset_fontsize = 9
fontsize_legend = 12
labelsize = 12
inset_labelsize = 9
linewidth = 2
s = 20
alpha = 0.3
marker = "o"
nb_instances = 1
bins = 20
xlim_bipartite = [0.488, 0.5126]
xlim_SBM = [0.6, 0.614559]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

file_SBM = "synch_predictions/graphs/SBM/SBM_instances/" \
                 "SBM_p11_0_7_p22_0_5_pout_0_2/" \
                 "2020_02_25_19h25min51sec_normalized_frobenius_norm_list"
with open(f'C:/Users/thivi/Documents/GitHub/network-synch/'
          f'{file_SBM}.json') as json_data:
    normalized_frobenius_norm_list_SBM = json.load(json_data)
file_bipartite = "synch_predictions/graphs/SBM/SBM_instances/" \
           "SBM_p11_0_p22_0_pout_0_2/" \
           "2020_02_25_21h03min22sec_normalized_frobenius_norm_list"
with open(f'C:/Users/thivi/Documents/GitHub/network-synch/'
          f'{file_bipartite}.json') as json_data:
    normalized_frobenius_norm_list_bipartite = json.load(json_data)

setup_label_list = ["Winfree$_{}$", "Kuramoto$_{}$", "theta$_{}$",
                    "Winfree$_{}$", "Kuramoto$_{}$", "theta$_{}$"]

rmse_naive_bipartite_winfree = [0.10980052, 0.08684932, 0.1147652 ,
                                0.11970248, 0.10294378, 0.11120733,
                                0.11994769, 0.10063658, 0.11357215,
                                0.0717049 , 0.12329655, 0.09871728,
                                0.05831169, 0.1215955 , 0.08394789,
                                0.12624013, 0.0714606 , 0.10593794,
                                0.10943834, 0.10288996, 0.10092367,
                                0.07833063, 0.10080149, 0.08553174,
                                0.0864269 , 0.09788222, 0.10624688,
                                0.10511687, 0.11720747, 0.08663625,
                                0.12124438, 0.10624306, 0.06517656,
                                0.079441  , 0.08725816, 0.12194087,
                                0.10408074, 0.10123387, 0.11448891,
                                0.07871697, 0.07719156, 0.07565594,
                                0.09675204, 0.09099431, 0.09334056,
                                0.1041469 , 0.13498121, 0.09975076,
                                0.0829915 , 0.08222384]

rmse_naive_SBM_winfree = [0.04075167, 0.0299169 , 0.03457754,
                          0.03970409, 0.03292419, 0.03542613,
                          0.03498113, 0.03087911, 0.03577121,
                          0.03380285, 0.03879658, 0.03392717,
                          0.03623088, 0.03592671, 0.04059287,
                          0.04271072, 0.03289911, 0.04979585,
                          0.03887469, 0.03395117, 0.03528987,
                          0.0369377 , 0.03165408, 0.03477215,
                          0.0297732 , 0.03864004, 0.03476782,
                          0.03317194, 0.04778918, 0.03635308,
                          0.04557293, 0.03925361, 0.03913806,
                          0.03481529, 0.03428456, 0.03679469,
                          0.03441249, 0.03081222, 0.03509989,
                          0.03787305, 0.03753732, 0.03385697,
                          0.02753609, 0.03421371, 0.03847003,
                          0.03571933, 0.03989032, 0.04050604,
                          0.03000987, 0.03539915]

fig = plt.figure(figsize=(6, 4))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=1, colspan=2)
plt.title("(a) Bipartite", fontsize=fontsize)
x = normalized_frobenius_norm_list_bipartite
weights = np.ones_like(x) / float(len(x))
ax1.hist(x, bins=bins, color=reduced_first_community_color, edgecolor="white",
         linewidth=1, weights=weights)
plt.text(0.492, 0.09, "$P(d)$", fontsize=fontsize)
# plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.ylabel("Count", fontsize=fontsize)
# ax1.set_xlim(xlim_bipartite)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# ax1.spines['left'].set_visible(False)
# ax1.set_yticklabels([])
# ax1.set_xticklabels([])
plt.axis('off')


ax2 = plt.subplot2grid((4, 4), (0, 2), rowspan=1, colspan=2)
plt.title("(b) SBM", fontsize=fontsize)
x = normalized_frobenius_norm_list_SBM
weights = np.ones_like(x) / float(len(x))
ax2.hist(x, bins=bins, color=reduced_first_community_color, edgecolor="white",
         linewidth=1, weights=weights)
plt.text(0.603, 0.09, "$P(d)$", fontsize=fontsize)
# plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.ylabel("Count", fontsize=fontsize)
# ax2.set_xlim(xlim_SBM)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax2.set_yticklabels([])
# ax2.set_xticklabels([])
plt.axis('off')


ax3 = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=2)
# plt.title("(a) Bipartite", fontsize=fontsize)
for i in [0, 1, 2]:
    ax3.scatter(frobenius_norm_matrix[i, :], rmse_transition_matrix[i, :],
                s=s, marker=marker, color=colors[i])
    # , label=setup_label_list[i])
# ax3.scatter(frobenius_norm_matrix[0, :], rmse_naive_bipartite_winfree,
#             s=s-10, zorder=10, color="r")
plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.xlabel("Normalized Fr$\\ddot{o}$benius norm \n"
#            "$d = || A - \\langle A \\rangle ||_{NF}$", fontsize=fontsize)
# plt.xlabel("Normalized Frobenius norm \n"
#            "$d = || A - \\langle A \\rangle ||_{NF}$", fontsize=fontsize)
plt.xlabel("$d$", fontsize=fontsize)
# plt.ylabel("RMSE($R_{complete}$, $R_{reduced}$)", fontsize=fontsize)
# plt.ylabel("RMSE($\\mathbf{R}^c$, $\\mathbf{R}^r$)", fontsize=fontsize)
# plt.ylabel("RMSE \ncomplete vs. reduced", fontsize=fontsize)
plt.ylabel("RMSE", fontsize=fontsize)
plt.xlim(xlim_bipartite)
# plt.ylim([-0.02, 0.4])
plt.xticks([0.49, 0.5, 0.51])
plt.yticks([0, 0.2, 0.4])

ax = plt.subplot2grid((4, 4), (1, 2), rowspan=3, colspan=2)
# plt.title("(b) SBM", fontsize=fontsize)
for j, i in enumerate([3, 4, 5]):
    ax.scatter(frobenius_norm_matrix[i, :], rmse_transition_matrix[i, :],
                s=s, marker=marker, label=setup_label_list[i], color=colors[j])
# ax.scatter(frobenius_norm_matrix[3, :], rmse_naive_SBM_winfree,
#            s=s-10, zorder=10, color="r")
plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.xlabel("Normalized Frobenius norm \n"
#            "$d = || A - \\langle A \\rangle ||_{NF}$", fontsize=fontsize)
plt.xlabel("$d$", fontsize=fontsize)
# plt.ylabel("RMSE($R_{complete}$, $R_{reduced}$)", fontsize=fontsize)
# plt.tight_layout()
# plt.legend(loc="best")
plt.xlim(xlim_SBM)
# plt.ylim([-0.02, 0.2])
plt.xticks([0.602, 0.607, 0.612])
plt.yticks([0, 0.1, 0.2])

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc=(0.2, 0.9),
                    ncol=3, fontsize=fontsize_legend)
h = legend.legendHandles
t = legend.texts
renderer = fig.canvas.get_renderer()

for i in range(len(h)):
    hbbox = h[i].get_window_extent(renderer)   # bounding box of handle
    tbbox = t[i].get_window_extent(renderer)   # bounding box of text

    x = tbbox.x0  # keep default horizontal position
    y = (hbbox.height - tbbox.height) / 3. + hbbox.y0  # vertically center the
    # bbox of the text to the bbox of the handle.

    t[i].set_position((x, y))  # set new position of the text


plt.subplots_adjust(
    top=0.805,
    bottom=0.165,
    left=0.135,
    right=0.98,
    hspace=0.0,
    wspace=0.745)

plt.show()
