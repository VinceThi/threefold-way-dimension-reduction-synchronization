from synch_predictions.plots.plot_complete_vs_reduced import *
from synch_predictions.simulations.data_synchro_transition_kuramoto import *
import matplotlib.pyplot as plt
import numpy as np
# import tkinter.simpledialog
# from tkinter import messagebox
# import time
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes


first_community_color = "#2171b5"
second_community_color = "#f16913"
reduced_first_community_color = "#9ecae1"
reduced_second_community_color = "#fdd0a2"
reduced_third_community_color = "#a1d99b"
reduced_fourth_community_color = "#9e9ac8"
total_color = "#525252"
fontsize = 12
inset_fontsize = 9
fontsize_legend = 12
labelsize = 12
inset_labelsize = 9
linewidth = 2
s = 30
alpha_plot = 0.5
marker = "."
x_lim_kb = [0.78, 2.52]
y_lim_kb = [0, 1.1]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

theoretical = 0
complete_flower_state = 0
complete_vs_reduced = 0

# file = "2020_02_04_16h36min05sec_alpha_0_7_data_R_dictionary" \
#        "_kuramoto_sakaguchi_star_2D"
# file = "2020_02_04_18h53min00sec_alpha_0_3pi_data_R_dictionary" \
#        "_kuramoto_sakaguchi_star_2D"
# file = "2020_02_04_21h45min55sec_alpha_0_3pi_moresigmapoints" \
#        "_data_R_dictionary_kuramoto_sakaguchi_star_2D"

# file = "2020_02_17_21h43min43sec_alpha_m0_1pi_data_R_dictionary" \
#        "_kuramoto_sakaguchi_star_2D"
# file = "2020_02_17_23h23min33sec_alpha_m0_2pi_data_R_dictionary" \
#        "_kuramoto_sakaguchi_star_2D"
file = "2020_02_18_04h17min10sec_alpha_m0_2pi_data_R_dictionary" \
       "_kuramoto_sakaguchi_star_2D"

with open(f'data/kuramoto_sakaguchi/{file}.json')\
        as json_data:
    R_dictionary = json.load(json_data)

r02f = 'r_vs_t_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_0.2'
# psi
phi02f = 'phi_vs_t_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_0.2'
r42f = 'r_vs_t_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_4.2'
# psi
phi42f = 'phi_vs_t_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_4.2'

R1f = 'R_vs_t_red_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_1'
Phi1f = 'Phi_vs_t_red_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_1'
R4f = 'R_vs_t_red_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_4'
Phi4f = 'Phi_vs_t_red_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_4'

with open(f'data/kuramoto_sakaguchi/{r02f}.json') as json_data:
    r02 = json.load(json_data)
with open(f'data/kuramoto_sakaguchi/{phi02f}.json') as json_data:
    phi02 = json.load(json_data)
with open(f'data/kuramoto_sakaguchi/{r42f}.json') as json_data:
    r42 = json.load(json_data)
with open(f'data/kuramoto_sakaguchi/{phi42f}.json') as json_data:
    phi42 = json.load(json_data)

with open(f'data/kuramoto_sakaguchi/{R1f}.json') as json_data:
    R1 = json.load(json_data)
with open(f'data/kuramoto_sakaguchi/{Phi1f}.json') as json_data:
    Phi1 = json.load(json_data)
with open(f'data/kuramoto_sakaguchi/{R4f}.json') as json_data:
    R4 = json.load(json_data)
with open(f'data/kuramoto_sakaguchi/{Phi4f}.json') as json_data:
    Phi4 = json.load(json_data)

R1_equilibrium = R1[-1]
R4_equilibrium = R4[-1]
Phi1_equilibrium = Phi1[-1]
Phi4_equilibrium = Phi4[-1]

r02_mean = np.mean(r02[len(r02)//2:])
r42_mean = np.mean(r42[len(r42)//2:])

sigma_array = np.array(R_dictionary["sigma_array"])
omega_array = np.array(R_dictionary["omega_array"])
alpha = np.array(R_dictionary["alpha"])
N = np.array(R_dictionary["N"])
nb_sigma = len(sigma_array)

r_s_matrix = np.array(R_dictionary["r"]).T
R_s_matrix = np.array(R_dictionary["R"]).T


def R_glob(Rp, Phi, N):
    return np.sqrt(1 + (N-1)**2*Rp**2 + 2*(N-1)*Rp*np.cos(Phi))/N


def Phi_fixed_point_1(omega1, omega2, sigma, N, alpha):
    return np.arcsin((omega1 - omega2) /
                     (sigma * np.sqrt(N**2 - 4*(N-1)*np.sin(alpha)**2)))\
           - np.arcsin(((N-2)*np.sin(alpha)) /
                       (np.sqrt(N**2 - 4*(N-1)*np.sin(alpha)**2)))


def R_top_branch(omega1, omega2, sigma, N, alpha):
    return R_glob(1, Phi_fixed_point_1(omega1, omega2, sigma, N, alpha), N)


def sigma_critical(omega1, omega2, N, alpha):
    return (omega1 - omega2)/np.sqrt(N**2 - 4*(N-1)*np.sin(alpha)**2)


def sigma_sc_minus(omega1, omega2, N, alpha):
    """
    Chen 2017
    :param omega1:
    :param omega2:
    :param N:
    :param alpha:
    :return:
    """
    return (omega1 - omega2)/((N-1)*np.cos(2*alpha) + 1)

# Equiv to sigma_sc_minus
# def sigma_b(omega1, omega2, N, alpha):
#     """ Huang 2016 """
#     return (omega1 - omega2)/(2*(N-1)*np.sin(alpha)**2 - N)


def sigma_cf(omega1, omega2, N, alpha):
    return (omega1 - omega2)/np.sqrt(2*(N-1)*np.cos(2*alpha) + 1)


def sigma_2(omega1, omega2, N, alpha):
    return (omega1 - omega2)/np.sqrt((N-1)**2 + 2*(N-1)*np.cos(2*alpha) + 1)


Rp_fixed_point_1 = 1
sig_c = sigma_critical(omega_array[0], omega_array[1], N, alpha)
R_c = R_top_branch(omega_array[0], omega_array[1], sig_c, N, alpha)
sig_sc_minus = sigma_sc_minus(omega_array[0], omega_array[1], N, alpha)
# sig_b = sigma_b(omega_array[0], omega_array[1], N, alpha)
# print(sig_b)
sig_cf = sigma_cf(omega_array[0], omega_array[1], N, alpha)
print(sig_sc_minus, sig_c, sig_cf)

sigma_array_Rc = np.linspace(0, sig_c, 1000)
sigma_array_fp1 = np.linspace(1.67, sigma_array[-1], 10000)
sigma_array_fp12 = np.linspace(sig_c, 1.67, 10000)


if theoretical:
    plt.figure(figsize=(6, 3))

    plt.subplot(121)
    sigma_array_a = np.linspace(9/11+0.002, 2.5, 1000000)
    alpha_array = [-np.pi/2, -1, 0, 1, np.pi/2]
    ax = plt.gca()
    for alpha in alpha_array:
        plt.plot(sigma_array_a, R_top_branch(10, 1, sigma_array_a, 11, alpha),
                 linewidth=linewidth, linestyle="-",
                 label="$\\alpha = {}$".format(np.round(alpha, 2)))
        sigma_critical_value = sigma_critical(10, 1, 11, alpha)
        R_critical = R_top_branch(10, 1, sigma_critical_value, 11, alpha)
        plt.scatter(sigma_critical_value, R_critical, s=60)
    plt.legend(loc=4, fontsize=fontsize_legend)
    plt.xlabel("$\\sigma$", fontsize=fontsize)
    ylab = plt.ylabel("$R^*$", fontsize=fontsize, labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    ylab.set_rotation(0)
    plt.ylim([0.8, 1.02])

    plt.subplot(122)
    # sigma_array_a = [9/11+0.002, 0.9, 1, 1.1, 2]
    sigma_array_a = [9/11+0.002, 1, 2]
    alpha_array = np.linspace(-np.pi/2, np.pi/2, 100000)
    ax = plt.gca()
    for sigma in sigma_array_a:
        plt.plot(alpha_array, R_top_branch(10, 1, sigma, 11, alpha_array),
                 linewidth=linewidth, linestyle="-",
                 label="$\\sigma = {}$".format(np.round(sigma, 2)))
    plt.legend(loc=4, fontsize=fontsize_legend)
    plt.xlabel("$\\alpha$", fontsize=fontsize)
    ylab = plt.ylabel("$R^*$", fontsize=fontsize, labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    ylab.set_rotation(0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    plt.ylim([0.8, 1.02])

    plt.tight_layout()
    plt.show()


elif complete_flower_state:
    r02f1 = f'r_vs_t_kuramoto_sakaguchi_star_2D_N_3_alpha_0_2pi_sigma_0.2'
    phi02f1 = f'phi_vs_t_kuramoto_sakaguchi_star_2D_N_3_alpha_0_2pi_sigma_0.2'
    r02f4 = f'r_vs_t_kuramoto_sakaguchi_star_2D_N_9_alpha_0_2pi_sigma_0.2'
    phi02f4 = f'phi_vs_t_kuramoto_sakaguchi_star_2D_N_9_alpha_0_2pi_sigma_0.2'
# r02f3 = f'r_vs_t_kuramoto_sakaguchi_star_2D_N_15_alpha_0_2pi_sigma_0.2'
# phi02f3 = f'phi_vs_t_kuramoto_sakaguchi_star_2D_N_15_alpha_0_2pi_sigma_0.2'
# r02f4 = f'r_vs_t_kuramoto_sakaguchi_star_2D_N_21_alpha_0_2pi_sigma_0.2'
# phi02f4 = f'phi_vs_t_kuramoto_sakaguchi_star_2D_N_21_alpha_0_2pi_sigma_0.2'
    r02f2 = f'r_vs_t_kuramoto_sakaguchi_star_2D_N_5_alpha_0_2pi_sigma_0.2'
    phi02f2 = f'phi_vs_t_kuramoto_sakaguchi_star_2D_N_5_alpha_0_2pi_sigma_0.2'
    r02f3 = f'r_vs_t_kuramoto_sakaguchi_star_2D_N_7_alpha_0_2pi_sigma_0.2'
    phi02f3 = f'phi_vs_t_kuramoto_sakaguchi_star_2D_N_7_alpha_0_2pi_sigma_0.2'
    with open(f'data/kuramoto_sakaguchi/{r02f1}.json') as json_data:
        r021 = json.load(json_data)
    with open(f'data/kuramoto_sakaguchi/{phi02f1}.json') as json_data:
        phi021 = json.load(json_data)
    with open(f'data/kuramoto_sakaguchi/{r02f2}.json') as json_data:
        r022 = json.load(json_data)
    with open(f'data/kuramoto_sakaguchi/{phi02f2}.json') as json_data:
        phi022 = json.load(json_data)
    with open(f'data/kuramoto_sakaguchi/{r02f3}.json') as json_data:
        r023 = json.load(json_data)
    with open(f'data/kuramoto_sakaguchi/{phi02f3}.json') as json_data:
        phi023 = json.load(json_data)
    with open(f'data/kuramoto_sakaguchi/{r02f4}.json') as json_data:
        r024 = json.load(json_data)
    with open(f'data/kuramoto_sakaguchi/{phi02f4}.json') as json_data:
        phi024 = json.load(json_data)

    color_flower = "#454545"

    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(221)
    ax1.set_title("(a) $N = 3$", fontsize=fontsize, y=1.02)
    ax1.plot(r021*np.cos(phi021), r021*np.sin(phi021), linewidth=0.2,
             color=color_flower)
    ax1.set_ylabel("$R(t) \: \\sin\\Psi(t)$", fontsize=fontsize)
    # ax1.set_xlabel("$R(t) \: \\cos\\Psi(t)$", fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)
    ax1.set_xlim([-0.75, 0.75])
    ax1.set_ylim([-0.75, 0.75])
    ax1.set_xticks([-0.7, 0, 0.7])
    ax1.set_yticks([-0.7, 0, 0.7])

    ax2 = plt.subplot(222)
    ax2.set_title("(b) $N = 5$", fontsize=fontsize, y=1.02)
    ax2.plot(r022 * np.cos(phi022), r022 * np.sin(phi022), linewidth=0.2,
             color=color_flower)
    # ax2.set_ylabel("$R(t) \: \\sin\\Psi(t)$", fontsize=fontsize)
    # ax2.set_xlabel("$R(t) \: \\cos\\Psi(t)$", fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    # ax2.set_xlim([-0.25, 0.25])
    # ax2.set_ylim([-0.25, 0.25])
    # ax2.set_xticks([-0.2, 0, 0.2])
    # ax2.set_yticks([-0.2, 0, 0.2])
    ax2.set_xlim([-0.45, 0.45])
    ax2.set_ylim([-0.45, 0.45])
    ax2.set_xticks([-0.4, 0, 0.4])
    ax2.set_yticks([-0.4, 0, 0.4])

    ax3 = plt.subplot(223)
    ax3.set_title("(c) $N = 7$", fontsize=fontsize, y=1.02)
    ax3.plot(r023 * np.cos(phi023), r023 * np.sin(phi023), linewidth=0.2,
             color=color_flower)
    ax3.set_ylabel("$R(t) \: \\sin\\Psi(t)$", fontsize=fontsize)
    ax3.set_xlabel("$R(t) \: \\cos\\Psi(t)$", fontsize=fontsize)
    ax3.tick_params(axis='both', which='major', labelsize=labelsize)
    ax3.set_xlim([-0.32, 0.32])
    ax3.set_ylim([-0.32, 0.32])
    ax3.set_xticks([-0.3, 0, 0.3])
    ax3.set_yticks([-0.3, 0, 0.3])

    ax4 = plt.subplot(224)
    ax4.set_title("(d) $N = 9$", fontsize=fontsize, y=1.02)
    ax4.plot(r024 * np.cos(phi024), r024 * np.sin(phi024), linewidth=0.2,
             color=color_flower)
    # ax4.set_ylabel("$R(t) \: \\sin\\Psi(t)$", fontsize=fontsize)
    ax4.set_xlabel("$R(t) \: \\cos\\Psi(t)$", fontsize=fontsize)
    ax4.tick_params(axis='both', which='major', labelsize=labelsize)
    # ax4.set_xlim([-0.11, 0.11])
    # ax4.set_ylim([-0.11, 0.11])
    # ax4.set_xticks([-0.1, 0, 0.1])
    # ax4.set_yticks([-0.1, 0, 0.1])
    ax4.set_xlim([-0.25, 0.25])
    ax4.set_ylim([-0.25, 0.25])
    ax4.set_xticks([-0.2, 0, 0.2])
    ax4.set_yticks([-0.2, 0, 0.2])

    plt.tight_layout()

    plt.show()


elif complete_vs_reduced:
    t0, t1, dt = 0, 100, 0.001
    r = 'r_vs_t_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_1'
    phi = 'phi_vs_t_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_1'
    R = 'R_vs_t_red_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_1'
    Phi = 'Phi_vs_t_red_kuramoto_sakaguchi_star_2D_N_11_alpha_0_2pi_sigma_1'

    with open(f'data/kuramoto_sakaguchi/{r}.json') as json_data:
        r = json.load(json_data)
    with open(f'data/kuramoto_sakaguchi/{phi}.json') as json_data:
        phi = json.load(json_data)
    with open(f'data/kuramoto_sakaguchi/{R}.json') as json_data:
        R = json.load(json_data)
    with open(f'data/kuramoto_sakaguchi/{Phi}.json') as json_data:
        Phi = json.load(json_data)

    first_community_color = "#2171b5"
    # second_community_color = "#f16913"
    reduced_first_community_color = "#9ecae1"
    # reduced_second_community_color = "#fdd0a2"
    total_color = "#525252"

    fig = plt.figure(figsize=(5, 5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    time_array = np.arange(t0, t1, dt)

    ax = plt.subplot(221)
    ax.plot(time_array, r, first_community_color,
             label="Complete dynamics")
    ax.plot(time_array, R, reduced_first_community_color,
             label="Reduced dynamics")
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("$R(t)$", fontsize=12)
    ax.set_xlim([96, 100])
    ax.set_ylim([0.117, 0.163])
    ax.set_xticks([96, 98, 100])
    plt.yticks([0.12, 0.14, 0.16])
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    # plt.legend(bbox_to_anchor=(1, 1, 1, 1),
    #            ncol=2, fontsize=fontsize_legend)

    # plt.subplot(224)
    # plt.plot(psi)
    # plt.xlabel("$t$", fontsize=12)
    # plt.ylabel("$\\Psi(t)$", fontsize=12)

    plt.subplot(222)
    plt.plot(time_array, np.sin(phi), first_community_color)
    plt.plot(time_array, np.sin(Phi), reduced_first_community_color)
    plt.xlabel("Time $t$", fontsize=12)
    plt.ylabel("$\\sin\\Phi(t)$", fontsize=12)
    plt.xlim([96, 100])
    plt.ylim([-1.1, 1.1])
    ax.set_xticks([96, 98, 100])
    plt.yticks([-1, 0, 1])
    plt.tick_params(axis='both', which='major', labelsize=labelsize)

    plt.subplot(223)
    plt.plot(r * np.cos(phi), r * np.sin(phi), linewidth=1,
             color=first_community_color)
    plt.xlabel("$R(t) \\cos\\Phi(t)$", fontsize=12)
    plt.ylabel("$R(t) \\sin\\Phi(t)$", fontsize=12)
    plt.xticks([-0.2, 0, 0.2])
    plt.yticks([-0.2, 0, 0.2])
    plt.tick_params(axis='both', which='major', labelsize=labelsize)

    plt.subplot(224)
    plt.plot(R * np.cos(Phi), R * np.sin(Phi), linewidth=1,
             color=reduced_first_community_color, zorder=0)
    plt.scatter(R[-1] * np.cos(Phi[-1]), R[-1] * np.sin(Phi[-1]), s=100,
                color=total_color, zorder=1)
    plt.xlabel("$R(t) \\cos\\Phi(t)$", fontsize=12)
    plt.ylabel("$R(t) \\sin\\Phi(t)$", fontsize=12)
    plt.xticks([-0.2, 0, 0.2])
    plt.yticks([-0.2, 0, 0.2])
    plt.tick_params(axis='both', which='major', labelsize=labelsize)

    # plt.subplot(222)
    # theta_core = kuramoto_sol[:, 0]
    # for i in range(N):
    #     plt.plot(np.cos(theta_core - kuramoto_sol[:, i]))
    # plt.xlabel("$t$", fontsize=12)
    # plt.ylabel("$\\cos(\\theta^c - \\theta_j^p)$", fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.14, 0.9),
               ncol=2, fontsize=fontsize_legend)
    # plt.tight_layout()
    plt.subplots_adjust(top=0.87,
                        bottom=0.11,
                        left=0.14,
                        right=0.97,
                        hspace=0.445,
                        wspace=0.525)
    plt.show()


else:
    fig = plt.figure(figsize=(5, 6))

    """ --------------------------(a)---------------------------------------"""
    ax1 = plt.subplot2grid((6, 4), (0, 0), rowspan=3, colspan=4)
    ax1.set_title("(a)", fontsize=fontsize, y=1.02)
    colors_complete = ["#252525", "#2171b5"]
    colors_reduced = ["#969696", reduced_first_community_color]
    color_flower = "#454545"

    for j, sigma in enumerate(sigma_array):
        if sigma > 1.67:
            zorder = [1, 0, 3, 2]
        else:
            zorder = [0, 1, 2, 3]
        for i in range(2):
            ax1.scatter(sigma, r_s_matrix[i][j], color=colors_complete[i],
                        marker=marker, s=s+200, linewidth=linewidth,
                        zorder=zorder[i])

            ax1.scatter(sigma, R_s_matrix[i][j], color=colors_reduced[i],
                        marker=marker, s=s, linewidth=linewidth,
                        zorder=zorder[i+2])
    ax1.vlines(1.663, 0.25, 0.95, linestyles="-",
               color=colors_complete[1], linewidth=linewidth - 0.5)
    ax1.vlines(1.673, 0.15, 0.95, linestyles="--",
               color=colors_reduced[1], linewidth=linewidth - 0.5)
    ax1.vlines(1.663, 0, 0.15, linestyles=(0, (1, 1)),
               color="#ababab", linewidth=linewidth)

    ax1.vlines(sig_c, R_c, 1.02, linestyles=(0, (1, 1)),
               color="#ababab", linewidth=linewidth)
    ax1.vlines(sig_cf, 0, 0.40, linestyles=(0, (1, 1)),
               color="#ababab", linewidth=linewidth)
    ax1.scatter(1.663, 0.66,  color=colors_complete[1],
                marker="v", s=s+70, zorder=10)
    ax1.scatter(sig_cf, 0.66, color=colors_complete[0],
                marker="^", s=s+70, zorder=10)
    ax1.scatter(1.673, 0.66,  color=colors_reduced[1],
                marker="v", s=s, zorder=11)
    ax1.scatter(sig_cf, 0.66, color=colors_reduced[0],
                marker="^", s=s, zorder=11)

    ax1.vlines(sig_cf, 0.40, 0.98, linestyles="-",
               color=colors_complete[0], linewidth=linewidth-0.5)
    ax1.vlines(sig_cf, 0.40, 0.98, linestyles="--",
               color=colors_reduced[0], linewidth=linewidth - 0.5)

    # plt.vlines(sig_sc_minus, 0, 1.02, linestyles="--",
    #            color="#ababab", linewidth=linewidth - 0.5)

    ax1.plot(sigma_array_fp1, R_top_branch(10, 1, sigma_array_fp1, 11, alpha),
             linewidth=linewidth-0.5, color=second_community_color,
             linestyle="-", zorder=4)
    ax1.plot(sigma_array_fp12, R_top_branch(10, 1,
                                            sigma_array_fp12, 11, alpha),
             linewidth=linewidth-0.5, color=second_community_color,
             linestyle="--", zorder=4)
    ax1.plot(sigma_array_Rc, R_c*np.ones(len(sigma_array_Rc)),
             linewidth=linewidth, color="#ababab",
             linestyle=(0, (1, 1)), zorder=5)

    # ax1.scatter(0.2, r02_mean, marker="*", s=300,
    #             color=color_flower,
    #             zorder=20, edgecolors='w')
    # ax1.scatter(4.2, r42_mean, marker="s", s=150,
    #             color=reduced_second_community_color,
    #             zorder=20, edgecolors='#525252')

    ax1.scatter(1, R1_equilibrium, marker="*", s=300,
                color=color_flower,
                zorder=20, edgecolors='w')
    ax1.scatter(4, R4_equilibrium, marker="s", s=150,
                color=reduced_second_community_color,
                zorder=20, edgecolors='#525252')

    left_bottom_axis_settings(ax1, "$\\sigma$", "$\\langle R \\rangle_t$",
                              x_lim_kb, y_lim_kb,
                              (0.5, -0.15), (-0.15, 0.45), fontsize,
                              labelsize, linewidth)
    # ax1.text(0.6, 0.52, r"\begin{center}Desynchronized \end{center}",
    #          fontsize=fontsize)
    # ax1.text(2, 0.52,   r"\begin{center}Coexistence\\ region \end{center}",
    #          fontsize=fontsize)
    # ax1.text(3.7, 0.52, r"\begin{center}Synchronized \end{center}",
    #          fontsize=fontsize)
    ax1.text(sig_c-0.05, 1.03, "$\\sigma_c$", fontsize=fontsize)
    ax1.text(-0.3, R_c-0.05, "$R^*_c$", fontsize=fontsize)
    ax1.text(1.663-0.05, -0.1, "$\\sigma_b$", fontsize=fontsize)
    ax1.text(sig_cf-0.05, -0.1, "$\\sigma_f$", fontsize=fontsize)
    ax1.set_xlim([sigma_array[0], sigma_array[-1]])
    ax1.set_yticks([0, 0.5, 1])

    """ --------------------------(b)---------------------------------------"""
    ax2 = plt.subplot2grid((6, 4), (4, 0), rowspan=3, colspan=2)
    ax2.set_title("(b)", fontsize=fontsize, y=1.02)
    # ax2.plot(r02 * np.cos(phi02), r02 * np.sin(phi02), linewidth=0.2,
    #          color=color_flower)
    # ax2.scatter(-0.17, 0.165, marker="*", s=300,
    #             color=color_flower, edgecolors='w')
    # ax2.set_ylabel("$R(t) \: \\sin\\Psi(t)$", fontsize=fontsize)
    # ax2.set_xlabel("$R(t) \: \\cos\\Psi(t)$", fontsize=fontsize)
    ax2.plot(R1 * np.cos(Phi1), R1 * np.sin(Phi1), linewidth=1,
             zorder=0,
             color=reduced_first_community_color)
    ax2.scatter(R1_equilibrium * np.cos(Phi1_equilibrium),
                R1_equilibrium * np.sin(Phi1_equilibrium),
                marker="*", s=300, zorder=1,
                color=color_flower, edgecolors='w')
    ax2.scatter(R1[200] * np.cos(Phi1[200]), R1[200] * np.sin(Phi1[200]),
                marker=(3, 0, 60), s=60, zorder=0,
                color=reduced_first_community_color)
    ax2.scatter(R1[3800] * np.cos(Phi1[3800]), R1[3800] * np.sin(Phi1[3800]),
                marker=(3, 0, 70), s=60, zorder=0,
                color=reduced_first_community_color)
    ax2.set_ylabel("$R(t) \: \\sin\\Phi(t)$", fontsize=fontsize)
    ax2.set_xlabel("$R(t) \: \\cos\\Phi(t)$", fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    # ax2.set_xlim([-0.2, 0.2])
    # ax2.set_ylim([-0.2, 0.2])
    ax2.set_xticks([0, 0.2])
    ax2.set_yticks([0, 0.2])
    ax2.set_xlim([-0.1, 0.25])
    ax2.set_ylim([-0.15, 0.22])

    """ --------------------------(c)---------------------------------------"""
    ax3 = plt.subplot2grid((6, 4), (4, 2), rowspan=3, colspan=2)
    ax3.set_title("(c)", fontsize=fontsize, y=1.02)
    # ax3.plot(r42*np.cos(phi42), r42*np.sin(phi42), linewidth=0.6,
    #          color=reduced_second_community_color)
    # ax3.scatter(-0.85, 0.85, marker="s", s=150,
    #             color=reduced_second_community_color, edgecolors='#525252')
    # ax3.set_xlabel("$R(t) \: \\cos\\Psi(t)$", fontsize=fontsize)
    ax3.plot(np.cos(np.linspace(0, 2*np.pi, 1000)),
             np.sin(np.linspace(0, 2*np.pi, 1000)), linewidth=1,
             zorder=0, linestyle=':',
             color="#252525")
    ax3.plot(R4 * np.cos(Phi4), R4 * np.sin(Phi4), linewidth=1.5,
             zorder=0,
             color=colors_reduced[0])
    ax3.scatter(R4_equilibrium * np.cos(Phi4_equilibrium),
                R4_equilibrium * np.sin(Phi4_equilibrium),
                marker="s", s=150, zorder=1,
                color=reduced_second_community_color, edgecolors='#525252')
    ax3.scatter(R4[1500] * np.cos(Phi4[1500]), R4[1500] * np.sin(Phi4[1500]),
                marker=(3, 0, 65), s=60, zorder=0,
                color=colors_reduced[0])

    ax3.set_xlabel("$R(t) \: \\cos\\Phi(t)$", fontsize=fontsize)
    ax3.tick_params(axis='both', which='major', labelsize=labelsize)
    # ax3.set_xlim([-1.02, 1.02])
    # ax3.set_ylim([-1.02, 1.02])
    ax3.set_xticks([-1, 0, 1])
    ax3.set_yticks([-1, 0, 1])
    # ax3.set_xlim([-0.15, 1.05])
    # ax3.set_ylim([-0.18, 1.05])
    ax3.set_xlim([-1.05, 1.05])
    ax3.set_ylim([-1.05, 1.05])

    plt.tight_layout()

    plt.subplots_adjust(hspace=0)

    plt.show()
    # if messagebox.askyesno("Python",
    #                        "Would you like to save the plot ?"):
    #     window = tkinter.Tk()
    #     window.withdraw()  # hides the window
    #   file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
    #     timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    #     # fig.savefig("data/kuramoto/"
    #     #             "{}_{}_complete_vs_reduced_kuramoto_star_2D"
    #     #             ".png".format(timestr, file))
    #     fig.savefig("data/kuramoto_sakaguchi/"
    #                 "{}_{}_complete_vs_reduced_kuramoto_star_2D"
    #                 ".pdf".format(timestr, file))
