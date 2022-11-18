import json
import matplotlib.pyplot as plt
from synch_predictions.graphs.special_graphs import mean_SBM
from synch_predictions.graphs.SBM_geometry import *
from tqdm import tqdm
import time

simulate = False
distance_list = []
file = "SBM_p11_0_7_p22_0_5_pout_0_2"
info_file = "2020_02_25_17h02min49sec_SBM_realizations" \
            "_info_dictionary_SBM_p11_0.7_p22_0.5_p12_0.2_N1_150_N2_100"
# file = "SBM_p11_0_p22_0_pout_0_2"
# info_file = "2020_02_25_17h17min22sec_SBM_realizations_info" \
#             "_dictionary_bipartite_p11_0_p22_0_p12_0.2_N1_150_N2_100"


if simulate:
    with open(f'SBM_instances/{file}/{info_file}.json') as json_data:
        info_dictionary = json.load(json_data)

    sizes = info_dictionary["sizes"]
    pq = info_dictionary["affinity_matrix"]
    number_realizations = info_dictionary["number_realizations"]

    mean_A = mean_SBM(sizes, pq)
    dmax = np.zeros(np.shape(mean_A))
    if pq[0][0] != 0:
        p_mat_up = max([pq[0][0], 1 - pq[0][0]])*np.ones((sizes[0], sizes[0]))
        p_mat_low = max([pq[1][1], 1 - pq[1][1]])*np.ones((sizes[1], sizes[1]))
        q_mat_up = max([pq[0][1], 1 - pq[0][1]])*np.ones((sizes[0], sizes[1]))
        q_mat_low = max([pq[1][0], 1 - pq[1][0]])*np.ones((sizes[1], sizes[0]))
    else:
        p_mat_up = np.zeros((sizes[0], sizes[0]))
        p_mat_low = np.zeros((sizes[1], sizes[1]))
        q_mat_up = max([pq[0][1], 1 - pq[0][1]])*np.ones((sizes[0], sizes[1]))
        q_mat_low = max([pq[1][0], 1 - pq[1][0]])*np.ones((sizes[1], sizes[0]))

    A_max = np.block([[p_mat_up, q_mat_up], [q_mat_low, p_mat_low]])
    norm_A_max = np.linalg.norm(A_max)

    normalized_frobenius_norm_list = []
    probability_list = []
    for i in tqdm(range(number_realizations)):

        with open(f'SBM_instances/{file}/A{i}.json') as json_data:
            A = np.array(json.load(json_data))

        normalized_frobenius_norm_list.append(
            np.linalg.norm(A - mean_A)/norm_A_max)

        # probability_list.append(get_probability_SBM(A, sizes, pq))
        # print((A, sizes, pq))

    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    with open(f'SBM_instances/{file}/'                                     
              f'{timestr}_normalized_frobenius_norm_list.json', 'w')\
            as outfile:
        json.dump(normalized_frobenius_norm_list, outfile)

    # with open(f'SBM_instances/{file}/'
    #           f'{timestr}_probability_list.json', 'w') as outfile:
    #     json.dump(probability_list, outfile)

else:
    filename = "2020_02_25_19h25min51sec_normalized_frobenius_norm_list"
    with open(f'SBM_instances/{file}/{filename}.json') as json_data:
        normalized_frobenius_norm_list = json.load(json_data)
    # filename = ""
    # with open(f'SBM_instances/{file}/{filename}.json') as json_data:
    #     probability_list = json.load(json_data)

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
nb_instances = 1
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure(figsize=(5.8, 3.2))
# plt.subplot(121)
x = normalized_frobenius_norm_list
weights = np.ones_like(x) / float(len(x))
plt.hist(x, bins=100, color=reduced_first_community_color, edgecolor="white",
         linewidth=1, weights=weights)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.xlabel("$|| A - \\langle A \\rangle ||_{NF}$", fontsize=fontsize)
plt.ylabel("Count", fontsize=fontsize)
# plt.tight_layout()
# plt.legend(loc="best")
# plt.xlim([67.8, 70.2])
# plt.ylim([-0.02, 0.4])
# plt.xticks([68, 69, 70])
# plt.yticks([0, 0.2, 0.4])
# plt.subplots_adjust(top=0.805, bottom=0.165, left=0.105, right=0.98,
#                     hspace=0.195, wspace=0.29)
# plt.subplot(122)
# x = probability_list
# weights = np.ones_like(x) / float(len(x))
# plt.hist(x, bins=100, color=reduced_first_community_color, edgecolor="white",
#          linewidth=1, weights=weights)
# plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.xlabel("$P(A)$", fontsize=fontsize)
# plt.ylabel("Count", fontsize=fontsize)
# plt.tight_layout()

plt.show()
