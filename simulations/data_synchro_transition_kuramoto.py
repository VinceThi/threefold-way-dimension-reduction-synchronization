import json
import numpy as np


"""                                                                            
    Kuramoto data                                                               
"""

# Bipartite
with open('data/kuramoto/2019_09_28_03h47min32sec_p_in_0_omega1_0.1_n1_300_n2_'
          '200_new_reduction_complete_r_matrix_kuramoto_2D.json') as json_data:
    r_kb_matrix = np.array(json.load(json_data))

with open('data/kuramoto/2019_09_28_03h47min32sec_p_in_0_omega1_0.1_n1_300_n2_'
          '200_new_reduction_complete_r1_matrix_kuramoto_2D.json') \
        as json_data:
    r1_kb_matrix = np.array(json.load(json_data))

with open('data/kuramoto/2019_09_28_03h47min32sec_p_in_0_omega1_0.1_n1_300_n2_'
          '200_new_reduction_complete_r2_matrix_kuramoto_2D.json') \
        as json_data:
    r2_kb_matrix = np.array(json.load(json_data))

with open('data/kuramoto/2019_09_28_03h47min32sec_p_in_0_omega1_0.1_n1_300_n2_'
          '200_new_reduction_reduced_R_matrix_kuramoto_2D.json') as json_data:
    R_kb_matrix = np.array(json.load(json_data))

with open('data/kuramoto/2019_09_28_03h47min32sec_p_in_0_omega1_0.1_n1_300_n2_'
          '200_new_reduction_reduced_R1_matrix_kuramoto_2D.json') as json_data:
    R1_kb_matrix = np.array(json.load(json_data))

with open('data/kuramoto/2019_09_28_03h47min32sec_p_in_0_omega1_0.1_n1_300_n2_'
          '200_new_reduction_reduced_R2_matrix_kuramoto_2D.json') as json_data:
    R2_kb_matrix = np.array(json.load(json_data))

# mean_r1_kb = np.mean(r1_kb_matrix, axis=0)
# mean_r2_kb = np.mean(r2_kb_matrix, axis=0)
# mean_R1_kb = np.mean(R1_kb_matrix, axis=0)
# mean_R2_kb = np.mean(R2_kb_matrix, axis=0)
# mean_r_kb = np.mean(r_kb_matrix, axis=0)
# mean_R_kb = np.mean(R_kb_matrix, axis=0)
#
# std_r1_kb = np.std(r1_kb_matrix, axis=0)
# std_r2_kb = np.std(r2_kb_matrix, axis=0)
# std_R1_kb = np.std(R1_kb_matrix, axis=0)
# std_R2_kb = np.std(R2_kb_matrix, axis=0)
# std_r_kb = np.std(r_kb_matrix, axis=0)
# std_R_kb = np.std(R_kb_matrix, axis=0)
""""""""""""""""""""""""

# SBM


with open('data/kuramoto/2019_07_18_12h00min05sec_p_in_0.5_omega1_0.1_n1_300_'
          'n2_200_article_complete_r_matrix_kuramoto_2D.json') as json_data:
    r_ks_matrix = np.array(json.load(json_data))


with open('data/kuramoto/2019_07_18_12h00min05sec_p_in_0.5_omega1_0.1_n1_300_'
          'n2_200_article_complete_r1_matrix_kuramoto_2D.json') as json_data:
    r1_ks_matrix = np.array(json.load(json_data))


with open('data/kuramoto/2019_07_18_12h00min05sec_p_in_0.5_omega1_0.1_n1_300_'
          'n2_200_article_complete_r2_matrix_kuramoto_2D.json') as json_data:
    r2_ks_matrix = np.array(json.load(json_data))


with open('data/kuramoto/2019_07_18_12h00min05sec_p_in_0.5_omega1_0.1_n1_300_'
          'n2_200_article_reduced_R_matrix_kuramoto_2D.json') as json_data:
    R_ks_matrix = np.array(json.load(json_data))


with open('data/kuramoto/2019_07_18_12h00min05sec_p_in_0.5_omega1_0.1_n1_300_'
          'n2_200_article_reduced_R1_matrix_kuramoto_2D.json') as json_data:
    R1_ks_matrix = np.array(json.load(json_data))


with open('data/kuramoto/2019_07_18_12h00min05sec_p_in_0.5_omega1_0.1_n1_300_'
          'n2_200_article_reduced_R2_matrix_kuramoto_2D.json') as json_data:
    R2_ks_matrix = np.array(json.load(json_data))

# mean_r1_ks = np.mean(r1_ks_matrix, axis=0)
# mean_r2_ks = np.mean(r2_ks_matrix, axis=0)
# mean_R1_ks = np.mean(R1_ks_matrix, axis=0)
# mean_R2_ks = np.mean(R2_ks_matrix, axis=0)
# mean_r_ks = np.mean(r_ks_matrix, axis=0)
# mean_R_ks = np.mean(R_ks_matrix, axis=0)
#
# std_r1_ks = np.std(r1_ks_matrix, axis=0)
# std_r2_ks = np.std(r2_ks_matrix, axis=0)
# std_R1_ks = np.std(R1_ks_matrix, axis=0)
# std_R2_ks = np.std(R2_ks_matrix, axis=0)
# std_r_ks = np.std(r_ks_matrix, axis=0)
# std_R_ks = np.std(R_ks_matrix, axis=0)
