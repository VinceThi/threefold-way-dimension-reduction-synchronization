import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import json
from tqdm import tqdm

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

# ------------------------- Numerical results ---------------------------------
simulate = 1


def kuramoto(phi, t, mu):
    return mu - np.sin(phi)


t0, t1, dt = 0, 500, 0.01
N = 2
plot_time_series = 1
phi0 = np.pi / 4
t = np.linspace(0, 100, 10000)
omega1 = -0.1
omega2 = 0.1
# sigma_array_exp = np.linspace(0.001, 0.5, 500)
sigma_array_exp = np.linspace(0.1, 0.3, 2)

r_time_averaged_array = np.zeros(len(sigma_array_exp))
for i, sigma in tqdm(enumerate(sigma_array_exp)):

    mu = (omega2 - omega1) / sigma
    sol = odeint(kuramoto, phi0, t, args=(mu,))
    if plot_time_series:
        plt.figure(figsize=(1, 1))
        plt.plot(t, np.cos(sol))
        ylab = plt.ylabel("$\\varphi$", fontsize=fontsize, labelpad=0)
        ylab.set_rotation(0)
        plt.xlabel("$t$", fontsize=fontsize, labelpad=0)
        plt.tight_layout()
        # plt.yticks([0, 0.5, 1])
        plt.ylim([-1, 1])
        plt.xlim([80, 100])
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        plt.show()

    r = (np.absolute(1 + np.exp(1j*sol))/2)
    r_mean = np.mean(r[5:])
    r_time_averaged_array[i] = r_mean

# timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
# with open(f'data/kuramoto/two_oscillators/'
#           f'{timestr}_R_vs_sigma_kuramoto_N2_'
#           f'w1_{omega1}_w2_{omega2}_A12_{1}_'
#           f'A21_{1}_'
#           f't0_{t0}_t1_{t1}_dt_{dt}'
#           f'.json', 'w') as outfile:
#     json.dump(r_time_averaged_array.tolist(), outfile)


# ----------------------- Theoretical results ---------------------------------

def mu(omega1, omega2, sigma):
    return (omega2 - omega1) / sigma


def R_transition(omega1, omega2, sigma):
    return np.sqrt((1 + np.sqrt(1-(mu(omega1, omega2, sigma))**2))/2)


def sigma_critical(omega1, omega2):
    return np.abs(omega2 - omega1)


# -------------------------- Plot results -------------------------------------
# if not simulate:
#     file = "2020_03_20_17h49min59sec_R_vs_sigma_kuramoto_N2_w1_" \
#            "-0.1_w2_0.1_A12_1_A21_1_averaging_9_t0_0_t1_1000_dt_0.05"
#
#     with open("data/kuramoto/two_oscillators/" + file + ".json") \
#             as json_data:
#         r_time_averaged_array = json.load(json_data)


# TODO faire deux inset: un désynchro, l'autre : phase-locked

plt.figure(figsize=(6, 3))
ax = plt.subplot(121)
sigma_array = np.linspace(0, 0.5, 1000000)
plt.plot(sigma_array_exp, r_time_averaged_array,
         linewidth=linewidth+2, linestyle="-", label="$\\langle R \\rangle_t$",
         zorder=0, color=first_community_color)
plt.plot(sigma_array, R_transition(omega1, omega2, sigma_array),
         linewidth=linewidth, linestyle="--", label="$R^*$", zorder=1,
         color=reduced_first_community_color)
plt.vlines(sigma_critical(omega1, omega2), 0, 1.02, linestyle="--",
           color="#bbbbbb", zorder=2)
# plt.text(0.1, 0.5, "Incohérents", fontsize=fontsize)
# plt.text(0.6, 0.5, "Synchronisés", fontsize=fontsize)
plt.text(sigma_critical(omega1, omega2)-0.025, 0.46, "$\\sigma_c$",
         fontsize=fontsize)
# sigma_critical_value = sigma_critical(10, 1, 11, alpha)
# R_critical = R_top_branch(10, 1, sigma_critical_value, 11, alpha)
# plt.scatter(sigma_critical_value, R_critical, s=60)
plt.legend(loc=4, fontsize=fontsize_legend)
plt.xlabel("$\\sigma$", fontsize=fontsize)
# ylab = plt.ylabel("$R^*$", fontsize=fontsize, labelpad=10)
# ylab.set_rotation(0)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(linewidth-1)
plt.ylim([0.5, 1.02])
plt.xlim([0, 0.5])
plt.xticks([0, 0.25, 0.5])
plt.yticks([0.5, 0.75, 1])

ax = plt.subplot(122)
omega1_array = np.zeros(1000)
omega2_array = np.linspace(-1, 1, 1000)
plt.plot(omega2_array-omega1_array, sigma_critical(omega1_array, omega2_array),
         linewidth=linewidth, linestyle="--", color="#bbbbbb")
plt.text(-0.7, 0.45, "$\\sigma_c$", fontsize=fontsize)
plt.text(0.55, 0.45, "$\\sigma_c$", fontsize=fontsize)
ylab = plt.ylabel("$\\sigma$", fontsize=fontsize, labelpad=10)
ylab.set_rotation(0)
plt.xlabel("$\\omega_2 - \omega_1$", fontsize=fontsize)
plt.tight_layout()
plt.yticks([0, 0.5, 1])
plt.ylim([0, 1])
plt.xlim([-1, 1])
plt.tick_params(axis='both', which='major', labelsize=labelsize)

plt.show()

