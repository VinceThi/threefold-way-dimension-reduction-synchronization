from synch_predictions.plots.plots_setup import *
import json


# CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
#                 "synch_predictions/graphs/two_triangles/CVM_data/" \
#                 "2020_10_09_17h12min56sec_CVM_dictionary_two_triangles" \
#                 "_2D_V_K_one_less_node_new_onmf_normalized_errors.json"

CVM_dict_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
                "synch_predictions/graphs/two_triangles/CVM_data/" \
                "2020_10_02_20h25min28sec_CVM_dictionary_two_triangles" \
                "_2D_normalized_errors.json"

with open(f'{CVM_dict_path}') as json_data:
    CVM_dict = json.load(json_data)

target_list = ["W", "WK", "WA", "WKA", "WAK",
               "K", "KW", "KA", "KWA", "KAW",
               "A", "AW", "AK", "AWK", "AKW"]

fig, axs = plt.subplots(3, 5, figsize=(8, 4.5))
axs = axs.ravel()
for i in range(len(target_list)):
    targets = target_list[i]
    if len(targets) == 1:
        T_1, T_2, T_3 = targets, "None", "None"
        title = f"${T_1}$"
    elif len(targets) == 2:
        T_1, T_2 = list(targets)
        T_3 = "None"
        title = f"${T_1} \\to {T_2}$"
    else:
        T_1, T_2, T_3 = list(targets)
        title = f"${T_1} \\to {T_2} \\to {T_3}$"
    M = CVM_dict["M_"+targets]
    axs[i].matshow(M, aspect="auto")
    axs[i].set_title(title, y=1.25)
    print(f"{targets}: M = {M}\n\n")
plt.tight_layout()
plt.show()
