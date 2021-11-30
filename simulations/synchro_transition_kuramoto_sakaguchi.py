from synch_predictions.dynamics.integrate import *
from synch_predictions.dynamics.dynamics import *
from synch_predictions.dynamics.reduced_dynamics import *
# from synch_predictions.graphs.graph_spectrum import *
# from synch_predictions.plots.plot_complete_vs_reduced import *
from synch_predictions.plots.plot_dynamics import *
import matplotlib
import numpy as np
# from numpy.linalg import multi_dot, pinv
import networkx as nx
import json
import tkinter.simpledialog
from tkinter import messagebox
import time


def get_synchro_transition_kuramoto_sakaguchi(p_array, sizes, nb_instances,
                                       t0, t1, dt, omega, alpha, sigma, N):
    import time
    from tqdm import tqdm

    r_avg_instances_list = []
    R_avg_instances_list = []

    for p in tqdm(p_array):
        r_list = []
        R_list = []
        for instance in range(nb_instances):
            time.sleep(3)
            pin = p
            pout = p
            pq = [[pin, pout], [pout, pin]]
            A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
            m = get_dominant_eigenvectors(A, 1)
            theta0 = 2*np.pi*np.random.rand(N)
            Z0 = np.sum(m*np.exp(1j*theta0))
            k_array = np.dot(A, np.ones(len(A[:, 0])).transpose())
            kappa = np.sum(m * k_array)
            hatkappa = np.sum(m**2*k_array)/np.sum(m**2)

            # Integrate complete dynamics
            args_complete_ks = (omega, sigma, alpha)
            ks_sol = integrate_dynamics(t0, t1, dt, kuramoto_sakaguchi, A,
                                        "vode", theta0,
                                        *args_complete_ks)
            rt = np.absolute(np.sum(m*np.exp(1j*ks_sol), axis=1))

            r_avg = np.sum(rt[9*int(t1//dt)//10:]) /\
                len(rt[9*int(t1//dt)//10:])
            r_list.append(r_avg)

            # Integrate recuced dynamics
            args_reduced_ks = (hatkappa, omega, sigma, alpha, N)
            Z_sol = integrate_dynamics(t0, t1, dt, reduced_kuramoto_sakaguchi,
                                       kappa, "zvode", Z0, *args_reduced_ks)
            Rt = np.absolute(Z_sol)
            R_avg = np.sum(Rt[9*int(t1//dt)//10:]) /\
                len(Rt[9*int(t1//dt)//10:])
            R_list.append(R_avg)

            # plt.plot(Rt)
            # plt.show()

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
t1 = 2000
dt = 0.01

# Structural parameter of the SBM
q = 1
sizes = [50, 50]
N = sum(sizes)
n1 = sizes[0]
n2 = sizes[1]

adjacency_mat = "SBM"

omega = 0
sigma = 1
alpha = 0.5

nb_instances = 10
p_array = np.linspace(0.001, 1, 10)

# Get result
r_list, R_list = get_synchro_transition_kuramoto_sakaguchi(
    p_array, sizes, nb_instances, t0, t1, dt, omega, alpha, sigma, N)


line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "adjacency_matrix_type = Erdos-Renyi\n"
line6 = "nb of graph and initial condition instances = {}" \
        "\n".format(nb_instances)
line7 = "p_array = {}\n".format(p_array)
line8 = "sigma = {}\n".format(sigma)
line9 = "omega = {}\n".format(omega)
line10 = "alpha = {}\n".format(alpha)
line11 = "theta0 = 2*np.pi*np.random.randn(N)\n"

# with open('synchro_data/theta_model/2019_04_09_13h06min23sec_theta_result'
#           '_complete_r_list.json') as json_data:
#     r_list = np.array(json.load(json_data))
# with open('synchro_data/theta_model/2019_04_09_13h06min23sec_theta_result'
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

    fig.savefig("data/theta/"
                "{}_{}_complete_vs_reduced_theta_model"
                ".pdf".format(timestr, file))

    f = open('data/theta/{}_{}.txt'.format(timestr, file), 'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, "\n", line8,
         line9, line10])

    f.close()

    with open('data/theta/{}_{}_complete_r_list'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(r_list, outfile)
    # with open('data/theta/{}_{}_reduced_R_list'
    #          '.json'.format(timestr, file), 'w') as outfile:
    #    json.dump(R_list, outfile)
