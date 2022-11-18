from dynamics.get_synchro_transition import *
from graphs.special_graphs import \
    two_triangles_graph_adjacency_matrix
from graphs.get_reduction_matrix_and_characteristics import *
from plots.plots_setup import *
# from synch_predictions.plots.plot_complete_vs_reduced import *

get_M_bool = 1
plot_time_series = 0
A = two_triangles_graph_adjacency_matrix()
graph_str = "two_triangles"
parameters_dictionary = {}
K = np.diag(np.sum(A, axis=1))
W = np.diag(np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2]))
L = K - A


if get_M_bool:
    # M_list_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
    #               "synch_predictions/graphs/two_triangles/" \
    #               "reduction_matrices_WKA/2020_09_30_17h46min13sec" \
    #               "_laplacian_freq_correct_one_M_list.json"
    M_list_path = "C:/Users/thivi/Documents/GitHub/network-synch/" \
                  "synch_predictions/graphs/two_triangles/" \
                  "reduction_matrices_WKA/2020_10_05_16h11min22sec" \
                  "_with_onmf_used_in_paper_M_list.json"
    with open(f'{M_list_path}') as json_data:
        M_array = np.array(json.load(json_data))

    index = 1   # in {0, 1, 2}
    M = M_array[index]
    target_choice_array = ["$L$", "$L \\to W$", "$W \\to L$"]

else:
    """--- n = 2 --- """
    # M = np.array([[4.23172418e-04, 4.78363499e-02, 0, 5.74273059e-01, 0,
    #                3.77467418e-01],
    #               [9.42736565e-01, 5.72631372e-02, 0, 2.98074889e-07, 0,
    #                0]])  # from V0, V2
    M = np.array([[2.02801153e-04, 4.89506739e-02, 0, 5.79364007e-01, 0,
                   3.71482518e-01],
                  [9.44190687e-01, 5.57937383e-02, 0, 1.55747148e-05, 0,
                   0]])  # V0, V2
    M = np. array([[5.64659203e-04, 9.04531588e-01, 9.49037530e-02,
                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                   [8.52527147e-01, 1.38856136e-04, 1.40736748e-15,
                    1.47333997e-01, 0.00000000e+00, 0.00000000e+00]])  # V4, V0

    """ --- n = 3 --- """
    # Pas d'erreur de ONMF dans cette matrice M ! Mais le 3e noeud n'est pas
    # impliqué dans la réduction.
    M = np.array([[0, 0, 0, 7.89803120e-001, 0, 2.10196880e-001],
                  [0, 9.59237960e-001, 0, 0, 4.07620398e-002, 0],
                  [1, 0, 0, 0, 0, 0]])  # from V0, V2, V1

    M = np.array([[9.99999894e-01, 1.03648185e-07, 2.29042667e-09,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                  [2.98285251e-14, 0.00000000e+00, 0.00000000e+00,
                   9.43199975e-01, 0.00000000e+00, 5.68000253e-02],
                  [9.08096040e-07, 9.69296904e-01, 3.07021883e-02,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

    target_choice = "X"

# Time parameters
t0, t1, dt = 0, 20000, 1   # 1000, 0.05
time_list = np.linspace(t0, t1, int(t1 / dt))
averaging = 5
sigma_array = np.array([0] + list(np.linspace(0.1, 4, 99)))
theta0 = np.array([7.87631282, -4.30630852, 11.02319849,
                   -2.33418242, -1.75137322, -0.57859127])
""" is drawn from the uniform distribution : 2*np.pi*np.random.randn(N) """

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

synchro_transition_data_dictionary = \
    get_synchro_transition_phase_dynamics_graphs(
        kuramoto, reduced_kuramoto_complex, t0, t1, dt, averaging,
        "A", "None", "None", theta0, M, sigma_array,
        W, A, plot_time_series=plot_time_series)

r = synchro_transition_data_dictionary["r"]
R = synchro_transition_data_dictionary["R"]

labelpad = 10
ylim = [0, 1.05]
xlim = [-0.05, 4.05]
xticks = [0, 2, 4]
yticks = [0, 0.5, 1.0]
x, y = 1.7, 0.02
fig = plt.figure(figsize=(3, 3))
ax = plt.subplot(111)
plt.plot(sigma_array, r, label="Complete", color=total_color,
         linewidth=linewidth)
plt.plot(sigma_array, R, label="Reduced", color=reduced_fourth_community_color,
         linewidth=linewidth)
plt.fill_between(sigma_array, r, R, color=total_color, alpha=0.2)
RMSE_transition = rmse(np.array(r), np.array(R))
if get_M_bool:
    plt.text(x=x, y=y, s=target_choice_array[index]+"\n"+"RMSE $\\approx$ {}"
             .format("%.3f" % np.round(RMSE_transition, 3)),
             fontsize=fontsize, zorder=15)
# left_bottom_axis_settings(ax, "$\\sigma$", "$\\langle R \\rangle_t$",
#                           (0.5, -0.35), (-0.3, 0.2), xlim, ylim, fontsize,
#                           labelsize, linewidth)
plt.xlabel("$\\sigma$", fontsize=fontsize)
ylab = plt.ylabel("$\\langle R \\rangle_t$", fontsize=fontsize,
                  labelpad=labelpad)
ylab.set_rotation(0)
plt.xlim(xlim)
plt.ylim(ylim)
plt.xticks(xticks)
plt.yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(linewidth-1)
# fig.subplots_adjust(bottom=0, top=0, right=0, left=0)
plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    fig.savefig(f"data/kuramoto/errors_kuramoto_combinations_WKA/"
                f"{timestr}_{file}_r_R_vs_sigma.pdf")
    fig.savefig(f"data/kuramoto/errors_kuramoto_combinations_WKA/"
                f"{timestr}_{file}_r_R_vs_sigma.png")

    with open(f'data/kuramoto/errors_kuramoto_combinations_WKA/'
              f'{timestr}_{file}_r_array'
              f'_complete_kuramoto.json', 'w')\
            as outfile:
        json.dump(r, outfile)
    with open(f'data/kuramoto/errors_kuramoto_combinations_WKA/'
              f'{timestr}_{file}_R_array'
              f'_reduced_kuramoto.json', 'w') as outfile:
        json.dump(R, outfile)

