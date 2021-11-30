from synch_predictions.plots.plots_setup import *
from synch_predictions.graphs.get_reduction_matrix_and_characteristics import *
from synch_predictions.plots.plot_spectrum import *
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

file_str = f'C:/Users/thivi/Documents/GitHub/network-synch/' \
           f'synch_predictions/simulations/data/wilson_cowan/' \
           f'bifurcation_diagram_adaptation/'

adaptation_rule_str = "BCM"

if adaptation_rule_str == "Hebb":
    bif_dig_str = "2021_01_23_13h27min31sec_50x50_grid_bifurcation_diagram" \
                  "_points_no_redundancy_reduced_wilson_cowan_Hebb"
    nb_eq_pts_str = "2021_01_23_13h27min31sec_50x50_grid_number_bifurcation" \
                    "_diagram_points_reduced_wilson_cowan_Hebb"
    parameters_str = "2021_01_23_13h27min31sec_50x50_grid_parameters" \
                     "_dictionary_wilson_cowan_Hebb"
elif adaptation_rule_str == "BCM":
    bif_dig_str = "2021_01_23_15h43min25sec_50x50grid_bifurcation_diagram" \
                  "_points_no_redundancy_reduced_wilson_cowan_BCM"
    nb_eq_pts_str = "2021_01_23_15h43min25sec_50x50_grid_number_bifurcation" \
                    "_diagram_points_reduced_wilson_cowan_BCM"
    parameters_str = "2021_01_23_15h43min25sec_50x50grid_parameters" \
                     "_dictionary_wilson_cowan_BCM"

with open(file_str + f'{bif_dig_str}.json') as json_data:
    bifurcation_diagram_points = np.array(json.load(json_data))
with open(file_str + f'{nb_eq_pts_str}.json') as json_data:
    number_bifurcation_diagram_points = np.array(json.load(json_data))
with open(file_str + f'{parameters_str}.json') as json_data:
    parameters_dictionary = json.load(json_data)

# When things will be done correctly ...
# b_linspace = parameters_dictionary["b_linspace"]
# c_linspace = parameters_dictionary["c_linspace"]
b_linspace = np.linspace(1, 5, 50)
c_linspace = np.linspace(0.01, 0.4, 50)

# number_bifurcation_diagram_points = \
#     np.zeros((len(b_linspace), len(c_linspace)))
# for i, b in enumerate(tqdm(b_linspace)):
#     for j, c in enumerate(tqdm(c_linspace)):
#         number_bifurcation_diagram_points[i, j] = \
#             len(bifurcation_diagram_points[i, j])

# with open(file_str + f'{}_50x50_grid_number'
#                      f'_bifurcation_diagram_points'
#                      f'_reduced_wilson_cowan_{adaptation_rule_str}.json',
#           'w') as outfile:
#     json.dump(number_bifurcation_diagram_points.tolist(), outfile)


fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
im = ax.imshow(number_bifurcation_diagram_points, cmap=cm4,
               interpolation='Kaiser', origin='lower', aspect="auto",
               extent=[0, c_linspace[-1],
                       b_linspace[0], b_linspace[-1]])
plt.xlabel('Time scales ratio', fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
ylab = plt.ylabel('Firing rate threshold', fontsize=fontsize, labelpad=5)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.08)
# cbar = fig.colorbar(im,
#                     cax=cax)  # ) # , shrink=1
# cbar.ax.set_title("Number of\n equilibrium \npoints",
#                   fontsize=fontsize, y=1.03)
# cbar.ax.tick_params(labelsize=labelsize)
plt.yticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.show()
if messagebox.askyesno("Python", "Would you like to save the plot ?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    file_str = f'C:/Users/thivi/Documents/GitHub/network-synch/' \
               f'synch_predictions/simulations/data/wilson_cowan/' \
               f'bifurcation_diagram_adaptation/{timestr}_{file}_'

    fig.savefig(file_str + f'bifurcation_diagram'
                           f'_wilson_cowan_{adaptation_rule_str}.pdf')
    fig.savefig(file_str + f'bifurcation_diagram'
                           f'_wilson_cowan_{adaptation_rule_str}.png')
