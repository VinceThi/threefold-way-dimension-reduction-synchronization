import json
import numpy as np

"""                                                                            
    Theta data                                                               
"""

# Erdos-Renyi - Excitatory coupling - sigma = 4

with open('data/theta/2019_07_03_17h09min10sec_sigma_4_I_1'
          '_complete_r_matrix.json') as json_data:
    r_Ip1_matrix = np.array(json.load(json_data))
with open('data/theta/2019_07_03_17h09min10sec_sigma_4_I_1'
          '_reduced_R_matrix.json') as json_data:
    R_Ip1_matrix = np.array(json.load(json_data))

with open('data/theta/2019_07_03_13h29min36sec_sigma_4_I_minus0_5'
          '_complete_r_matrix.json') as json_data:
    r_I05_matrix = np.array(json.load(json_data))
with open('data/theta/2019_07_03_13h29min36sec_sigma_4_I_minus0_5'
          '_reduced_R_matrix.json') as json_data:
    R_I05_matrix = np.array(json.load(json_data))

with open('data/theta/2019_07_15_19h34min38sec_Erdos_I_m1_50_instances'
          '_complete_r_matrix.json') as json_data:
    r_I1_matrix = np.array(json.load(json_data))
with open('data/theta/2019_07_15_19h34min38sec_Erdos_I_m1_50_instances'
          '_reduced_R_matrix.json') as json_data:
    R_I1_matrix = np.array(json.load(json_data))

with open('data/theta/2019_07_03_13h15min52sec_sigma_4_I_minus2'
          '_complete_r_matrix.json') as json_data:
    r_I2_matrix = np.array(json.load(json_data))
with open('data/theta/2019_07_03_13h15min52sec_sigma_4_I_minus2'
          '_reduced_R_matrix.json') as json_data:
    R_I2_matrix = np.array(json.load(json_data))

mean_r_Ip1 = np.mean(r_Ip1_matrix, axis=0)
mean_R_Ip1 = np.mean(R_Ip1_matrix, axis=0)
mean_r_I05 = np.mean(r_I05_matrix, axis=0)
mean_R_I05 = np.mean(R_I05_matrix, axis=0)
mean_r_I1 = np.mean(r_I1_matrix, axis=0)
mean_R_I1 = np.mean(R_I1_matrix, axis=0)
mean_r_I2 = np.mean(r_I2_matrix, axis=0)
mean_R_I2 = np.mean(R_I2_matrix, axis=0)

std_r_Ip1 = np.std(r_Ip1_matrix, axis=0)
std_R_Ip1 = np.std(R_Ip1_matrix, axis=0)
std_r_I05 = np.std(r_I05_matrix, axis=0)
std_R_I05 = np.std(R_I05_matrix, axis=0)
std_r_I1 = np.std(r_I1_matrix, axis=0)
std_R_I1 = np.std(R_I1_matrix, axis=0)
std_r_I2 = np.std(r_I2_matrix, axis=0)
std_R_I2 = np.std(R_I2_matrix, axis=0)


# SBM - Excitatory coupling - sigma = 4

with open('data/theta/2019_07_17_23h14min38sec_good_macroscopic_observable'
          '_complete_r_matrix_theta_2D.json') as json_data:
    r_ts_matrix = np.array(json.load(json_data))
with open('data/theta/2019_07_17_23h14min38sec_good_macroscopic_observable'
          '_reduced_R_matrix_theta_2D.json') as json_data:
    R_ts_matrix = np.array(json.load(json_data))

with open('data/theta/2019_07_17_23h14min38sec_good_macroscopic_observable'
          '_complete_r1_matrix_theta_2D.json') as json_data:
    r1_ts_matrix = np.array(json.load(json_data))
with open('data/theta/2019_07_17_23h14min38sec_good_macroscopic_observable'
          '_reduced_R1_matrix_theta_2D.json') as json_data:
    R1_ts_matrix = np.array(json.load(json_data))

with open('data/theta/2019_07_17_23h14min38sec_good_macroscopic_observable'
          '_complete_r2_matrix_theta_2D.json') as json_data:
    r2_ts_matrix = np.array(json.load(json_data))
with open('data/theta/2019_07_17_23h14min38sec_good_macroscopic_observable'
          '_reduced_R2_matrix_theta_2D.json') as json_data:
    R2_ts_matrix = np.array(json.load(json_data))
