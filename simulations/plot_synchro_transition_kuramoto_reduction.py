from dynamics.get_reduction_errors import *
# from plots.plots_setup import *
import matplotlib.pyplot as plt
import json
import numpy as np

first_community_color = "#2171b5"
second_community_color = "#f16913"
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
labelpad = 10
xlim = [-0.15, 8.2]
xticks = [0, 2, 4, 6, 8]
x, y = 0.02, 0.02
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

with open('data/kuramoto/kuramoto_secIIID_article/'
          '2020_01_20_01h19min00sec_test_data_reduction'
          '_parameters_dictionary_kuramoto.json'
          ) as json_data:
    reduction_data_dictionary = json.load(json_data)

sigma_array = np.array(reduction_data_dictionary["sigma_array"])
r_array = np.array(reduction_data_dictionary["r"])
R_array = np.array(reduction_data_dictionary["R"])
T_1 = reduction_data_dictionary["T_1"]
T_2 = reduction_data_dictionary["T_2"]
T_3 = reduction_data_dictionary["T_3"]
RMSE_transition = RMSE(r_array, R_array)


fig = plt.figure(figsize=(3, 3))

plt.plot(sigma_array, r_array, color=total_color,
         label="$\\langle R^{com} \\rangle_t$")
plt.plot(sigma_array, R_array,
         color=reduced_third_community_color, linestyle="-",
         label="$\\langle R^{red} \\rangle_t$")
plt.fill_between(sigma_array, r_array, R_array,
                 color=total_color, alpha=0.2)
plt.text(x=x, y=y, s="$T_1 = {}$ \n $T_2 = {}$\n $T_3 = {}$ \n"
                     "RMSE $\\approx$ {}"
         .format(T_1, T_2, T_3, "%.3f" % np.round(RMSE_transition, 3)),
         fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.xticks(xticks)
plt.ylim([0, 1.02])
plt.xlim(xlim)
# ylab = plt.ylabel("$\\langle R \\rangle_t$", fontsize=fontsize, labelpad=12)
# ylab.set_rotation(0)
plt.xlabel("$\\sigma$", fontsize=fontsize)
plt.legend(loc=4, fontsize=fontsize_legend, handlelength=0.9)
plt.tight_layout()

plt.show()
