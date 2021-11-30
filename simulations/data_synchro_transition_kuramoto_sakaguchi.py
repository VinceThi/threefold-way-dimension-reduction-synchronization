import json
import numpy as np


with open('data/kuramoto_sakaguchi/2018_10_17_21h44min26sec_verygood_prediction_mean'
          '_Rp1_complete_vs_alpha.json') as json_data:
    rp1_1 = np.array(json.load(json_data))

with open('data/kuramoto_sakaguchi/2018_10_17_21h44min26sec_verygood_prediction_mean'
          '_Rp2_complete_vs_alpha.json') as json_data:
    rp2_1 = np.array(json.load(json_data))

with open('data/kuramoto_sakaguchi/2018_10_17_21h44min26sec_verygood_prediction_mean'
          '_Rp1_reduced_vs_alpha.json') as json_data:
    Rp1_1 = np.array(json.load(json_data))

with open('data/kuramoto_sakaguchi/2018_10_17_21h44min26sec_verygood_prediction_mean'
          '_Rp2_reduced_vs_alpha.json') as json_data:
    Rp2_1 = np.array(json.load(json_data))

with open('data/kuramoto_sakaguchi/2018_10_17_23h12min52sec_transition_separation'
          '_mean_Rp1_complete_vs_alpha.json') as json_data:
    rp1_2 = np.array(json.load(json_data))

with open('data/kuramoto_sakaguchi/2018_10_17_23h12min52sec_transition_separation'
          '_mean_Rp2_complete_vs_alpha.json') as json_data:
    rp2_2 = np.array(json.load(json_data))

with open('data/kuramoto_sakaguchi/2018_10_17_23h12min52sec_transition_separation'
          '_mean_Rp1_reduced_vs_alpha.json') as json_data:
    Rp1_2 = np.array(json.load(json_data))

with open('data/kuramoto_sakaguchi/2018_10_17_23h12min52sec_transition_separation'
          '_mean_Rp2_reduced_vs_alpha.json') as json_data:
    Rp2_2 = np.array(json.load(json_data))
