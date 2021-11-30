from synch_predictions.dynamics.get_synchro_transition import *
import json
# import tkinter.simpledialog
# from tkinter import messagebox
import time

# Time parameters
t0, t1, dt = 0, 20000, 1   # 1000, 0.05
time_list = np.linspace(t0, t1, int(t1 / dt))

""" Two-triangles, n = 2 """
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_20_21h22min04sec_CVM_two_triangles_2D_dictionary' \
#                 '_dominant_V_A.json'
# sigma_array = np.linspace(0.01, 8, 500)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_15h12min47sec_CVM_two_triangles_2D_' \
#                 'dictionary_V_K_all_nodes.json'
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_18h41min12sec_CVM_two_triangles_2D_dictionary' \
#                 '_V_K_one_less_node.json'
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_18h51min35sec_CVM_two_triangles_2D_dictionary' \
#                 '_V_K_perturbed.json'
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_02_01_02h10min15sec_CVM_dictionary_two_triangles' \
#                 '_2D_snmf_and_onmf.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_02_04_20h01min19sec_CVM_dictionary_two_triangles' \
#                 '_2D_multiple_inits_V_K_perturbed.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_02_04_23h56min37sec_CVM_dictionary_two_triangles' \
#                 '_2D_multiple_inits_V_K_one_neglected_node.json'
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_06_02h29min09sec_CVM_dictionary_two_triangles" \
#                 "_2D_not_aligned.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_07_16h48min16sec_CVM_dictionary_two_triangles" \
#                 "_2D_not_aligned_2.json"
# sigma_array = np.linspace(0.1, 4, 100)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_06_01h54min15sec_CVM_dictionary_two_triangles" \
#                 "_2D_snmf_onmf_multiple_inits.json"
# sigma_array = np.linspace(0.1, 4, 100)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_06_17h02min18sec_CVM_dictionary_two_triangles" \
#                 "_2D_multiple_inits.json"
# sigma_array = np.array([0] + list(np.linspace(0.1, 4, 99)))
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_27_16h47min12sec_CVM_dictionary_two_triangles" \
#                 "_2D_V_K_all_nodes.json"
# sigma_array = np.array([0] + list(np.linspace(0.1, 4, 99)))
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_09_30_15h23min37sec_CVM_dictionary_two_triangles_2D" \
#                 "_different_algorithm_onmf.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_10_02_20h25min28sec_CVM_dictionary_two_triangles" \
#                 "_2D_normalized_errors.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_10_05_01h44min54sec_CVM_dictionary_two_triangles" \
#                 "_2D_V_K_one_less_node_new_onmf.json"
CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
                "synch_predictions/graphs/two_triangles/CVM_data/" \
                "2020_10_09_17h12min56sec_CVM_dictionary_two_triangles" \
                "_2D_V_K_one_less_node_new_onmf_normalized_errors.json"
sigma_array = np.array([0] + list(np.linspace(0.1, 4, 99)))  

""" Two-triangles, n = 3 """
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_20_21h22min31sec_CVM_two_triangles' \
#                 '_3D_dictionary_V0_V1_V3.json'
# sigma_array = np.linspace(0.01, 8, 500)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_01_29_19h58min26sec_CVM_two_triangles_3D' \
#                 '_dictionary_V0_V1_V3.json'
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/two_triangles/CVM_data/' \
#                 '2020_02_01_02h10min58sec_CVM_dictionary_two_triangles_3D' \
#                 '_snmf_and_onmf.json'
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_06_01h58min10sec_CVM_dictionary_two_triangles" \
#                 "_3D_snmf_onmf_multiple_inits.json"
# sigma_array = np.linspace(0.01, 8, 100)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_10_16h45min42sec_CVM_dictionary_two_triangles_3D" \
#                 "_V_K_one_less_node.json"
# sigma_array = np.linspace(0.1, 4, 100)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_28_13h51min30sec_CVM_dictionary_two_triangles" \
#                 "_3D_V_K_all_nodes_V_A_dominant_veps.json"
# sigma_array = np.array([0] + list(np.linspace(0.1, 4, 99)))
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_28_18h31min13sec_CVM_dictionary_two_triangles_3D" \
#                 "_V_K_all_nodes_V_A_V0_V1_V3.json"
# sigma_array = np.array([0] + list(np.linspace(0.1, 4, 99)))
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_02_07_17h15min14sec_CVM_dictionary_two_triangles" \
#                 "_3D_multiple_inits.json"
# sigma_array = np.array([0] + list(np.linspace(0.1, 4, 99)))
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_10_04_15h49min25sec_CVM_dictionary_two_triangles_3D" \
#                 "_different_algorithm_onmf_normalized_errors.json"
# sigma_array = np.array([0] + list(np.linspace(0.1, 4, 99)))
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_10_04_20h35min26sec_CVM_dictionary_two_triangles" \
#                 "_3D_V_A_is_V0_V1_V2_new_onmf.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_10_05_00h48min43sec_CVM_dictionary_two_triangles" \
#                 "_3D_V_A_V0_V1_V2_V_K_full_new_onmf.json"
# sigma_array = np.array([0] + list(np.linspace(0.1, 4, 99)))

""" Small bipartite, n = 2 """
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_21_16h33min34sec_CVM_small_bipartite_2D'  \
#                 '_dictionary_V_K_neglects_node_1.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_21_22h05min41sec_CVM_small_bipartite_2D' \
#                 '_dictionary_dominant_veps.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_23_14h41min07sec_CVM_small_bipartite_2D' \
#                 '_dictionary_deg_freq.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_23_15h11min37sec_CVM_small_bipartite_2D' \
#                 '_dictionary_other_deg_freq.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_29_23h36min49sec_CVM_dictionary_' \
#                 'small_bipartite_2D_dom_eig.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_02_01_00h19min29sec_CVM_dictionary_small_bipartite' \
#                 '_2D_snmf_and_onmf.json'
# sigma_array = np.linspace(0.1, 4, 100)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/small_bipartite/CVM_data/" \
#                 "2020_02_10_18h37min26sec_CVM_dictionary_small_bipartite_2D"\
#                 "_freq_not_aligned.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/small_bipartite/CVM_data/" \
#                 "2020_02_10_19h16min35sec_CVM_dictionary_small_bipartite_2D"\
#                 "_freq_not_aligned_2.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/small_bipartite/CVM_data/" \
#                 "2020_02_10_19h26min27sec_CVM_dictionary_small_bipartite_2D"\
#                 "_freq_not_aligned_3.json"
# sigma_array = np.linspace(0.1, 3, 50)

""" Small bipartite, n = 3 """
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_21_21h49min46sec_CVM_small_bipartite_3D' \
#                 '_dictionary_with_eigA_0.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_21_21h56min08sec_CVM_small_bipartite_3D' \
#                 '_dictionary_dominant_veps.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_23_15h06min19sec_CVM_small_bipartite_3D' \
#                 '_dictionary_deg_freq.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_23_15h12min31sec_CVM_small_bipartite_3D' \
#                 '_dictionary_other_deg_freq.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_01_30_00h04min39sec_CVM_dictionary_small_bipartite' \
#                 '_3D_V0_V2_V1_dom_eig.json'
# CVM_dict_path = 'C:/Users/thivi/Documents/GitHub/network-synch/' \
#                 'synch_predictions/graphs/small_bipartite/CVM_data/' \
#                 '2020_02_01_00h33min30sec_CVM_dictionary_small_bipartite' \
#                 '_3D_snmf_and_onmf.json'
# sigma_array = np.linspace(0.1, 4, 100)

""" Bipartite, n = 2 """
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/SBM/CVM_data/" \
#                 "2020_02_09_15h40min53sec_CVM_dictionary_bipartite" \
#                 "_2D_bipartite_pout_0_2.json"
# sigma_array = np.linspace(2, 5, 50)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/SBM/CVM_data/" \
#                 "2020_02_10_08h37min08sec_CVM_dictionary" \
#                 "_bipartite_2D_pout_0_5.json"
# sigma_array = np.linspace(0.1, 4, 10)
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/SBM/CVM_data/" \
#                 "2020_02_10_10h34min28sec_CVM_dictionary_bipartite_2D" \
#                 "_no_variation_freq_pout_0_2.json"
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/SBM/CVM_data/" \
#                 "2020_02_10_11h25min46sec_CVM_dictionary_bipartite_2D" \
#                 "_pout_0_2_omega1_0_1.json"
#
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/SBM/CVM_data/" \
#                 "2020_02_11_22h08min21sec_CVM_dictionary_bipartite_2D" \
#                 "_pout_0_2_omega1_0_1_random.json"
# graph_str = "bipartite"
#

""" SBM, n = 2 """
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/SBM/CVM_data/" \
#                 "2020_02_09_23h06min16sec_CVM_dictionary_SBM_2D" \
#                 "_pin_0_8_pout_0_2.json"
# sigma_array = np.linspace(2, 5, 50)

""" Bipartite, n = 3 """
# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/SBM/CVM_data/" \
#                 "2020_02_10_09h14min33sec_CVM_dictionary_bipartite_3D" \
#                 "_pout_0_2.json"
#
# sigma_array = np.linspace(2, 5, 50)


with open(f'{CVM_dict_path}') as json_data:
    CVM_dict = json.load(json_data)

A = np.array(CVM_dict["A"])
# print(np.diag(np.array(CVM_dict["W"])))
N = int(len(A[0]))
n = len(np.array(CVM_dict["M_W"])[:, 0])
theta0 = np.array([7.87631282, -4.30630852, 11.02319849,
                   -2.33418242, -1.75137322, -0.57859127])
""" is drawn from the uniform distribution : 2*np.pi*np.random.randn(N) """
# theta0 = np.linspace(0, 2*np.pi*(1-1/N), N) np.random.normal(0, np.pi, N)

targets_possibilities = ["W", "K", "A", "WK", "WA", "KW", "KA", "AW",
                         "AK", "WKA", "WAK", "KWA", "KAW", "AWK", "AKW"]

multiple_synchro_transition_dictionary = \
    get_multiple_synchro_transition_phase_dynamics_graphs(
            kuramoto, reduced_kuramoto_complex, t0, t1, dt, 5,
            CVM_dict, targets_possibilities, theta0, sigma_array,
            plot_time_series=False)
# get_multiple_synchro_transition_kuramoto_graphs(t0, t1, dt, 5,
#                                                 CVM_dict,
#                                                 theta0, sigma_array,
#                                                 plot_time_series=True)

line1 = f"t0 = {t0}\n"
line2 = f"t1 = {t1}\n"
line3 = f"deltat = {dt}\n"
line4 = f"Number of nodes : N = {len(A[0])}\n"
line5 = f"Dimension of the reduced dynamics : n = {n}\n"
# line6 = f"theta0 = np.array({np.round(theta0[0], 3)}," \
#         f" {np.round(theta0[-1], 3)}, {len(theta0)})\n"
line6 = f"theta0 = {theta0}\n"
line7 = f"sigma_array = np.array({np.round(sigma_array[0], 3)}," \
        f" {np.round(sigma_array[-1], 3)}, {len(sigma_array)})\n"
line8 = " multiple_synchro_transition_dictionary" \
        " is a dictionary that contains the transitions for all the \n " \
        " possible targets W,K,A,WK,WA,KW,KA,AW,AK,WKA,WAK,KWA,KAW,AWK,AKW\n" \
        " The keys are given below, where {} = all the possible targets: \n" \
        "  Keys                                                       \n " \
        "{ r_...,                     -> Global synchro observables  \n " \
        "                            of the complete dynamics        \n " \
        "  R_...,                     -> Global synchro observables  \n " \
        "                              of the reduced dynamics       \n " \
        "  r_mu_matrix,               -> Mesoscopic synchro observables  \n " \
        "                                of the complete dynamics        \n " \
        "  R_mu_matrix...,            -> Mesoscopic synchro observables  \n " \
        "                                of the reduced dynamics       \n" \
        "  sigma_array, W, K, A, n, N, theta0}\n\n"
line9 = f"CVM_dictionary path :\n {CVM_dict_path}"

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

# if messagebox.askyesno("Python",
#                        "Would you like to save the parameters, "
#                        "the data and the plot?"):
# window = tkinter.Tk()
# window.withdraw()  # hides the window
# file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

file = "two_triangles"

f = open(f'data/kuramoto/kuramoto_secIIID_article/'
         f'{timestr}_multiple_synchro_transition_dataset'
         f'_informations_kuramoto_{n}D_{file}'
         f'.txt', 'w')
f.writelines(
    [line1, line2, line3, line4, line5, line6, line7, "\n", line8, line9])

f.close()

with open(f'data/kuramoto/kuramoto_secIIID_article/'
          f'{timestr}_multiple_synchro_transition'
          f'_dictionary_kuramoto_{n}D_{file}'
          f'.json', 'w') as outfile:
    json.dump(multiple_synchro_transition_dictionary, outfile)
