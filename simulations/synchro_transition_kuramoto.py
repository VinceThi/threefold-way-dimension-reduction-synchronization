from dynamics.integrate import *
from dynamics.dynamics import *
from dynamics.reduced_dynamics import *
from graphs.graph_spectrum import *
# from plots.plot_complete_vs_reduced import *
from plots.plot_dynamics import *
import matplotlib
import numpy as np
from numpy.linalg import multi_dot, pinv
import networkx as nx
import json
import tkinter.simpledialog
from tkinter import messagebox
import time


def get_synchro_transition_kuramoto(p_array, sizes, nb_instances, t0, t1, dt,
                                    omega_mu, omega_std, sigma):
    import time
    from tqdm import tqdm

    r_avg_instances_list = []
    R_avg_instances_list = []

    for p in tqdm(p_array):
        r_list = []
        R_list = []
        for instance in range(nb_instances):
            time.sleep(3)

            # Graph parameters
            N = sum(sizes)
            pin = p
            pout = p
            pq = [[pin, pout], [pout, pin]]
            A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
            k_array = np.dot(A, np.ones(len(A[:, 0])).transpose())
            K = np.diag(k_array)
            V_not_normalized = get_eigenvectors_matrix(A, 1)
            V = V_not_normalized / np.sum(V_not_normalized[0])
            # V is a matrix 1xN.
            Vp = pinv(V)

            # Dynamical parameters
            omega = np.random.normal(omega_mu, omega_std, N)
            W = np.diag(omega)
            VW = np.array([omega * V[0]])
            VWp = pinv(VW)
            M = np.array([V[0], VW[0]])
            Mp = pinv(M)

            # Reduced paramters
            redomega = np.dot(V, omega)[0]
            redomega2 = np.dot(V, omega ** 2)[0]
            var_omega = redomega2 - redomega ** 2
            kappa = np.dot(V, k_array)[0]
            hatkappa = multi_dot([V, K, Vp])[0][0]
            primekappa = multi_dot([VW, K, VWp])[0][0]
            Q = multi_dot([V, W, A, Mp])
            tau = Q[0][0]
            eta = Q[0][1]

            # Initial conditions
            theta0 = 2 * np.pi * np.random.rand(N)
            z0 = np.exp(1j * theta0)
            Z0 = np.dot(V, z0)
            Z0_array = Z0 * np.ones(N)
            W0 = np.dot(VW, z0 - Z0_array)
            ZW0 = np.array([Z0, W0])

            # Integration complete dynamics
            args_kuramoto = (omega, sigma)
            kuramoto_sol = integrate_dynamics(t0, t1, dt, kuramoto, A,
                                              "vode", theta0, *args_kuramoto)
            rt_kuramoto = np.absolute(
                np.sum(V[0]*np.exp(1j*kuramoto_sol), axis=1))
            mean_r = np.sum(rt_kuramoto[9*int(t1//dt)//10:]) / \
                     len(rt_kuramoto[9*int(t1//dt)//10:])
            r_list.append(mean_r)

            # Integration reduced dynamics
            args_Z_kuramoto = (hatkappa, primekappa, var_omega, redomega,
                               tau, eta, sigma, N)
            Z_kuramoto_sol = integrate_dynamics(t0, t1, dt,
                                                reduced_kuramoto_freq_1D,
                                                kappa, "zvode", ZW0,
                                                *args_Z_kuramoto)
            Rt_kuramoto = np.absolute(Z_kuramoto_sol[:, 0])
            mean_R = np.sum(Rt_kuramoto[9*int(t1//dt)//10:]) / \
                     len(Rt_kuramoto[9*int(t1//dt)//10:])

            R_list.append(mean_R)

        r_avg_instances_list.append(np.sum(r_list)/nb_instances)
        R_avg_instances_list.append(np.sum(R_list)/nb_instances)

    return r_avg_instances_list, R_avg_instances_list


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
first_community_color = "#064878"   # 6, 72, 120
second_community_color = "#ff370f"  # 255, 55, 15
fontsize = 40
fontsize_legend = 30
labelsize = 30
linewidth = 3


# Time parameters
t0 = 0
t1 = 20
dt = 0.05

# Structural parameter of the SBM
q = 1


n1 = 500
n2 = 500
sizes = [n1, n2]
N = sum(sizes)


sigma = 1
omega_mu = 0
omega_std = 0.05


nb_instances = 20
p_array = np.linspace(0.01, 0.4, 10)

# Get result
r_list, R_list = get_synchro_transition_kuramoto(p_array, sizes, nb_instances,
                                                 t0, t1, dt, omega_mu,
                                                 omega_std, sigma)


line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "adjacency_matrix_type = Erdos-Renyi\n"
line6 = "nb of graph and initial condition instances = {}" \
        "\n".format(nb_instances)
line7 = "p_array = {}\n".format(p_array)
line8 = "sigma = {}\n".format(sigma)
line9 = "omega_distribution " \
        "= np.random.normal({}, {}, N)\n".format(omega_mu, omega_std)
line10 = "theta0 = 2*np.pi*np.random.randn(N)\n"

# with open('synchro_data/theta/2019_04_09_13h06min23sec_theta_result'
#           '_complete_r_list.json') as json_data:
#     r_list = np.array(json.load(json_data))
# with open('synchro_data/theta/2019_04_09_13h06min23sec_theta_result'
#           '_reduced_R_list.json') as json_data:
#     R_list = np.array(json.load(json_data))

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")


fig = plt.figure(figsize=(10, 8))
# plt.plot(p_array, r_list, color=first_community_color,
#          linestyle='-', linewidth=linewidth, label="$Theoretical$")
# plt.plot(p_array, R_list, color="lightblue",
#          linestyle='-', linewidth=linewidth, label='$Predicted$')
plt.scatter(p_array, r_list, s=100, color=first_community_color,
            label="$Complete$")
plt.scatter(p_array, R_list, s=50, color="lightblue",
            label="$Reduced$", marker='s')
plt.ylabel("$R$", fontsize=fontsize)
plt.xlabel("$p$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([0, 1.02])
plt.xlim([0, 1.02])
plt.legend(loc=4, fontsize=fontsize_legend)
# , ncol=2)# bbox_to_anchor=(1.35, 1.01),
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters, "
                       "the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    fig.savefig("data/kuramoto/"
                "{}_{}_complete_vs_reduced_kuramoto_model"
                ".pdf".format(timestr, file))

    f = open('data/kuramoto/{}_{}.txt'
             .format(timestr, file), 'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, "\n", line8,
         line9, line10])

    f.close()

    with open('data/kuramoto/{}_{}_complete_r_list'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(r_list, outfile)
    with open('data/kuramoto/'
              '{}_{}_reduced_R_list.json'.format(timestr, file), 'w') \
            as outfile:
        json.dump(R_list, outfile)

