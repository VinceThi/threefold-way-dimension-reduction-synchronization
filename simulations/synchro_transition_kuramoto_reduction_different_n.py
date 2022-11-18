from dynamics.get_synchro_transition import *
from graphs.special_graphs import \
    two_triangles_graph_adjacency_matrix
from graphs.get_reduction_matrix_and_characteristics\
    import rmse
from plots.plots_setup import *
import json
import time


simulate = False
show_errors = False

# Time parameters
t0, t1, dt = 0, 20000, 0.8   # 20000, 1    usually
time_list = np.linspace(t0, t1, int(t1 / dt))
averaging = 8  # 5, usually, 2020-10-04

theta0 = np.array([7.87631282, -4.30630852, 11.02319849,
                   -2.33418242, -1.75137322, -0.57859127])
""" is drawn from the uniform distribution : 2*np.pi*np.random.randn(N) """

# Define matrices
A = two_triangles_graph_adjacency_matrix()
K = np.diag(np.sum(A, axis=1))
N = len(A[0])
graph_str = "two_triangles"
sigma_array = np.array([0] + list(np.linspace(0.1, 2, 99)))
W = np.diag(np.array([0.1, 0.1, -0.2, -0.2, 0.1, 0.1]))

# SNMF niter = 500, ONMF maxiter = 500, nb_init = 5000
M_list_path = "2020_10_03_18h20min32sec_two_targets_A_W_M_list" \
              "_corrected_n2_and_n3.json"
# M_list_path = "2020_10_02_19h43min31sec_with_normalized_errors_M_list.json"
# "2020_09_28_13h13min42sec_snmf_niter_2000_onmf_max_iter_500" \
#           "_nb_init_1000_M_list.json"
# M_list_path = "2020_09_28_18h11min06sec_snmf_niter2000_onmf_maxiter1000" \
#               "_nbinit2000_M_list.json"
# M_list_path = "2020_09_28_18h11min06sec_snmf_niter2000_onmf_maxiter1000" \
#               "_nbinit2000_M_list.json"
with open(f'C:/Users/thivi/Documents/GitHub/network-synch/synch_predictions/'
          f'graphs/two_triangles/reduction_matrices_n/{M_list_path}')\
        as json_data:
    M_list = json.load(json_data)

""" Uniform partitions """
# M_list = [np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]]),
#           np.array([[1/3, 1/3, 1/3, 0, 0, 0],
#                     [0, 0, 0, 1/3, 1/3, 1/3]]),
#           np.array([[1/2, 1/2, 0, 0, 0, 0],
#                     [0, 0, 1/2, 1/2, 0, 0],
#                     [0, 0, 0, 0, 1/2, 1/2]]),
#           np.array([[1/2, 1/2, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 1/2, 1/2]]),
#           np.array([[1, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 1/2, 1/2]]),
#           np.eye(6, 6)]

if show_errors:
    RMSE_errors_compatibility_eq_W = []
    RMSE_errors_compatibility_eq_K = []
    RMSE_errors_compatibility_eq_A = []
    for M in M_list:
        RMSE_errors_compatibility_eq_W.append(get_rmse_error(M, W))
        RMSE_errors_compatibility_eq_K.append(get_rmse_error(M, K))
        RMSE_errors_compatibility_eq_A.append(get_rmse_error(M, A))
    plt.figure(figsize=(4, 3))
    ax = plt.subplot(111)
    ax.plot([1, 2, 3, 4, 5, 6],  RMSE_errors_compatibility_eq_W,
            label="RMSE$_W$",
            linewidth=linewidth, color=reduced_first_community_color)
    ax.plot([1, 2, 3, 4, 5, 6],  RMSE_errors_compatibility_eq_K,
            label="RMSE$_K$",
            linewidth=linewidth, color=reduced_second_community_color)
    ax.plot([1, 2, 3, 4, 5, 6],  RMSE_errors_compatibility_eq_A,
            label="RMSE$_A$",
            linewidth=linewidth, color=reduced_third_community_color)
    plt.legend(loc="best", fontsize=fontsize_legend)
    plt.xlabel("$n$", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth-1)
    plt.tight_layout()
    plt.show()

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

if simulate:

    complete_sol_list = get_solutions_complete_phase_dynamics_graphs(
        kuramoto, t0, t1, dt, theta0, sigma_array, W, A)

    r_array = np.zeros((6, len(sigma_array)))
    R_array = np.zeros((6, len(sigma_array)))
    for i in tqdm(range(6)):

        M = np.array(M_list[i])

        reduced_sol_list = get_solutions_reduced_phase_dynamics_graphs(
            reduced_kuramoto_complex, t0, t1, dt, theta0, M,
            sigma_array, W, A)

        # synchro_transition_data_dictionary = \
        #     get_synchro_transition_phase_dynamics_graphs(
        #         kuramoto, reduced_kuramoto_complex, t0, t1, dt, averaging,
        #        "A", "None", "None", theta0, np.array(M_list[i]), sigma_array,
        #         W, A, plot_time_series=False)

        r_array[i] = measure_synchronization_complete_phase_dynamics(
            complete_sol_list, sigma_array, t1, dt, averaging, M)[0]
        R_array[i] = measure_synchronization_reduced_phase_dynamics(
            reduced_sol_list, sigma_array, t1, dt, averaging, M)[0]

        print("\n\n")

else:
    # file_r = '2020_09_28_19h55min01sec_other_M_r_array_complete_kuramoto'
    # file_r = "2020_09_29_11h20min24sec_first_M_n3_" \
    #          "didnotneed_onmf_r_array_complete_kuramoto"
    file_r = "2020_10_04_16h43min03sec_two_targets_corrected" \
             "_n2_and_n3_r_array_complete_kuramoto"
    # file_r = "2020_10_05_20h52min23sec_with_uniform_partitions" \
    #          "_n_2_is_a_test_r_array_complete_kuramoto"
    with open("data/kuramoto/errors_kuramoto_vs_n/" + file_r + ".json") \
            as json_data:
        r_array = np.array(json.load(json_data))

    # file_R = '2020_09_28_19h55min01sec_other_M_R_array_reduced_kuramoto'
    # file_R = "2020_09_29_11h20min24sec_first_M_n3_" \
    #          "didnotneed_onmf_R_array_reduced_kuramoto"
    file_R = "2020_10_04_16h43min03sec_two_targets_corrected" \
             "_n2_and_n3_R_array_reduced_kuramoto"
    # file_R = "2020_10_05_20h52min23sec_with_uniform_partitions" \
    #          "_n_2_is_a_test_R_array_reduced_kuramoto"
    with open("data/kuramoto/errors_kuramoto_vs_n/" + file_R + ".json") \
            as json_data:
        R_array = np.array(json.load(json_data))


fig = plt.figure(figsize=(5, 3))
# color_list = ["#A1D99B", "#7DCB75", "#59BC4E",
#               "#449A3B", "#33742C", "#224D1D"]
x, y = -0.05, -0.05
ylim = [-0.1, 1.05]
xlim = [-0.1, sigma_array[-1]+0.1]
xticks = [0, 1, 2]
yticks = [0, 0.5, 1.0]
labelpad = 15
number_digits = 3

ax1 = plt.subplot(231)
plt.plot(sigma_array, r_array[0], color=complete_grey)
plt.plot(sigma_array, R_array[0], color=third_community_color)
plt.fill_between(sigma_array, r_array[0], R_array[0],
                 color=complete_grey, alpha=0.2)
ylab = plt.ylabel("$\\langle R \\rangle_t$",
                  fontsize=fontsize, labelpad=labelpad)
ylab.set_rotation(0)
RMSE_transition = rmse(np.array(r_array[0]), np.array(R_array[0]))
plt.text(x=x, y=y, s="$n = 1$" + "\n" + "RMSE $\\approx$ {}"
         .format("%.3f" % np.round(RMSE_transition, number_digits)),
         fontsize=fontsize-1, zorder=15)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xticks(xticks)
plt.yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
for axis in ['bottom', 'left']:
    ax1.spines[axis].set_linewidth(linewidth-1)


ax2 = plt.subplot(232)
plt.plot(sigma_array, r_array[1], color=complete_grey)
plt.plot(sigma_array, R_array[1], label=f"$n =$ {2}",
         color=third_community_color)
plt.fill_between(sigma_array, r_array[1], R_array[1],
                 color=complete_grey, alpha=0.2)
RMSE_transition = rmse(np.array(r_array[1]), np.array(R_array[1]))
plt.text(x=x, y=y, s="$n = 2$" + "\n" + "RMSE $\\approx$ {}"
         .format("%.3f" % np.round(RMSE_transition, number_digits)),
         fontsize=fontsize-1, zorder=15)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xticks(xticks)
plt.yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
for axis in ['bottom', 'left']:
    ax2.spines[axis].set_linewidth(linewidth-1)


ax3 = plt.subplot(233)
plt.plot(sigma_array, r_array[2], color=complete_grey)
plt.plot(sigma_array, R_array[2], label=f"$n =$ {3}",
         color=third_community_color)
plt.fill_between(sigma_array, r_array[2], R_array[2],
                 color=complete_grey, alpha=0.2)
RMSE_transition = rmse(np.array(r_array[2]), np.array(R_array[2]))
plt.text(x=x, y=y, s="$n = 3$" + "\n" + "RMSE $\\approx$ {}"
         .format("%.3f" % np.round(RMSE_transition, number_digits)),
         fontsize=fontsize-1, zorder=15)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xticks(xticks)
plt.yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')
for axis in ['bottom', 'left']:
    ax3.spines[axis].set_linewidth(linewidth-1)


ax4 = plt.subplot(234)
plt.plot(sigma_array, r_array[3], color=complete_grey)
plt.plot(sigma_array, R_array[3], label=f"$n =$ {4}",
         color=third_community_color)
plt.fill_between(sigma_array, r_array[3], R_array[3],
                 color=complete_grey, alpha=0.2)
ylab = plt.ylabel("$\\langle R \\rangle_t$",
                  fontsize=fontsize, labelpad=labelpad)
ylab.set_rotation(0)
plt.xlabel("$\\sigma$", fontsize=fontsize)
RMSE_transition = rmse(np.array(r_array[3]), np.array(R_array[3]))
plt.text(x=x, y=y, s="$n = 4$" + "\n" + "RMSE $\\approx$ {}"
         .format("%.3f" % np.round(RMSE_transition, number_digits)),
         fontsize=fontsize-1, zorder=15)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xticks(xticks)
plt.yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.yaxis.set_ticks_position('left')
ax4.xaxis.set_ticks_position('bottom')
for axis in ['bottom', 'left']:
    ax4.spines[axis].set_linewidth(linewidth-1)


ax5 = plt.subplot(235)
plt.plot(sigma_array, r_array[4], color=complete_grey)
plt.plot(sigma_array, R_array[4], label=f"$n =$ {5}",
         color=third_community_color)
plt.fill_between(sigma_array, r_array[4], R_array[4],
                 color=complete_grey, alpha=0.2)
plt.xlabel("$\\sigma$", fontsize=fontsize)
RMSE_transition = rmse(np.array(r_array[4]), np.array(R_array[4]))
plt.text(x=x, y=y, s="$n = 5$" + "\n" + "RMSE $\\approx$ {}"
         .format("%.3f" % np.round(RMSE_transition, number_digits)),
         fontsize=fontsize-1, zorder=15)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xticks(xticks)
plt.yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax5.yaxis.set_ticks_position('left')
ax5.xaxis.set_ticks_position('bottom')
for axis in ['bottom', 'left']:
    ax5.spines[axis].set_linewidth(linewidth-1)

ax6 = plt.subplot(236)
plt.plot(sigma_array, r_array[5], color=complete_grey)
plt.plot(sigma_array, R_array[5], label=f"$n =$ {6}",
         color=third_community_color)
plt.fill_between(sigma_array, r_array[5], R_array[5],
                 color=complete_grey, alpha=0.2)
plt.xlabel("$\\sigma$", fontsize=fontsize)
plt.text(x=x, y=y, s="$n = 6$" + "\n" + "RMSE = 0",
         fontsize=fontsize-1, zorder=15)
plt.ylim(ylim)
plt.xlim(xlim)
plt.xticks(xticks)
plt.yticks(yticks)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
ax6.spines['right'].set_visible(False)
ax6.spines['top'].set_visible(False)
ax6.yaxis.set_ticks_position('left')
ax6.xaxis.set_ticks_position('bottom')
for axis in ['bottom', 'left']:
    ax6.spines[axis].set_linewidth(linewidth-1)
plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters, "
                       "the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    fig.savefig(f"data/kuramoto/errors_kuramoto_vs_n/"
                f"{timestr}_{file}_errors_vs_n.pdf")
    fig.savefig(f"data/kuramoto/errors_kuramoto_vs_n/"
                f"{timestr}_{file}_errors_vs_n.png")

    if simulate:
        with open(f'data/kuramoto/errors_kuramoto_vs_n/{timestr}_{file}'
                  f'_r_array'
                  f'_complete_kuramoto.json', 'w')\
                as outfile:
            json.dump(r_array.tolist(), outfile)
        with open(f'data/kuramoto/errors_kuramoto_vs_n/{timestr}_{file}'
                  f'_R_array'
                  f'_reduced_kuramoto.json', 'w') as outfile:
            json.dump(R_array.tolist(), outfile)
