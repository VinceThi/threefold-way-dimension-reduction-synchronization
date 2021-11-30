import json
import numpy as np

"""
    Winfree data
"""

# Bipartite
with open('data/winfree/2019_09_28_03h40min29sec_p_in_0_omega1_0.3_n1_300_n2_'
          '200_new_reduction_complete_r_matrix_winfree_2D.json') as json_data:
    r_wb_matrix = np.array(json.load(json_data))

with open('data/winfree/2019_09_28_03h40min29sec_p_in_0_omega1_0.3_n1_300_n2_'
          '200_new_reduction_complete_r1_matrix_winfree_2D.json') as json_data:
    r1_wb_matrix = np.array(json.load(json_data))

with open('data/winfree/2019_09_28_03h40min29sec_p_in_0_omega1_0.3_n1_300_n2_'
          '200_new_reduction_complete_r2_matrix_winfree_2D.json') as json_data:
    r2_wb_matrix = np.array(json.load(json_data))

with open('data/winfree/2019_09_28_03h40min29sec_p_in_0_omega1_0.3_n1_300_n2_'
          '200_new_reduction_reduced_R_matrix_winfree_2D.json') as json_data:
    R_wb_matrix = np.array(json.load(json_data))

with open('data/winfree/2019_09_28_03h40min29sec_p_in_0_omega1_0.3_n1_300_n2_'
          '200_new_reduction_reduced_R1_matrix_winfree_2D.json') as json_data:
    R1_wb_matrix = np.array(json.load(json_data))

with open('data/winfree/2019_09_28_03h40min29sec_p_in_0_omega1_0.3_n1_300_n2_'
          '200_new_reduction_reduced_R2_matrix_winfree_2D.json') as json_data:
    R2_wb_matrix = np.array(json.load(json_data))

# mean_r1_wb = np.mean(r1_wb_matrix, axis=0)
# mean_r2_wb = np.mean(r2_wb_matrix, axis=0)
# mean_R1_wb = np.mean(R1_wb_matrix, axis=0)
# mean_R2_wb = np.mean(R2_wb_matrix, axis=0)
# mean_r_wb = np.mean(r_wb_matrix, axis=0)
# mean_R_wb = np.mean(R_wb_matrix, axis=0)
#
# std_r1_wb = np.std(r1_wb_matrix, axis=0)
# std_r2_wb = np.std(r2_wb_matrix, axis=0)
# std_R1_wb = np.std(R1_wb_matrix, axis=0)
# std_R2_wb = np.std(R2_wb_matrix, axis=0)
# std_r_wb = np.std(r_wb_matrix, axis=0)
# std_R_wb = np.std(R_wb_matrix, axis=0)


# SBM
with open('data/winfree/2019_07_18_05h49min05sec_p_in_0.5_omega1_0.3_n1_300_'
          'n2_200_article_complete_r_matrix_winfree_2D.json') as json_data:
    r_ws_matrix = np.array(json.load(json_data))

with open('data/winfree/2019_07_18_05h49min05sec_p_in_0.5_omega1_0.3_n1_300_'
          'n2_200_article_complete_r1_matrix_winfree_2D.json') as json_data:
    r1_ws_matrix = np.array(json.load(json_data))

with open('data/winfree/2019_07_18_05h49min05sec_p_in_0.5_omega1_0.3_n1_300_'
          'n2_200_article_complete_r2_matrix_winfree_2D.json') as json_data:
    r2_ws_matrix = np.array(json.load(json_data))

with open('data/winfree/2019_07_18_05h49min05sec_p_in_0.5_omega1_0.3_n1_300_'
          'n2_200_article_reduced_R_matrix_winfree_2D.json') as json_data:
    R_ws_matrix = np.array(json.load(json_data))

with open('data/winfree/2019_07_18_05h49min05sec_p_in_0.5_omega1_0.3_n1_300_'
          'n2_200_article_reduced_R1_matrix_winfree_2D.json') as json_data:
    R1_ws_matrix = np.array(json.load(json_data))

with open('data/winfree/2019_07_18_05h49min05sec_p_in_0.5_omega1_0.3_n1_300_'
          'n2_200_article_reduced_R2_matrix_winfree_2D.json') as json_data:
    R2_ws_matrix = np.array(json.load(json_data))

# mean_r1_ws = np.mean(r1_ws_matrix, axis=0)
# mean_r2_ws = np.mean(r2_ws_matrix, axis=0)
# mean_R1_ws = np.mean(R1_ws_matrix, axis=0)
# mean_R2_ws = np.mean(R2_ws_matrix, axis=0)
# mean_r_ws = np.mean(r_ws_matrix, axis=0)
# mean_R_ws = np.mean(R_ws_matrix, axis=0)
#
# std_r1_ws = np.std(r1_ws_matrix, axis=0)
# std_r2_ws = np.std(r2_ws_matrix, axis=0)
# std_R1_ws = np.std(R1_ws_matrix, axis=0)
# std_R2_ws = np.std(R2_ws_matrix, axis=0)
# std_r_ws = np.std(r_ws_matrix, axis=0)
# std_R_ws = np.std(R_ws_matrix, axis=0)
