from dynamics.get_synchro_transition import *
from graphs.special_graphs import \
    two_triangles_graph_adjacency_matrix
import json
import tkinter.simpledialog
from tkinter import messagebox
import time
import matplotlib.pyplot as plt

# Time parameters
t0, t1, dt = 0, 500, 0.05   # 1000, 0.05
time_list = np.linspace(t0, t1, int(t1 / dt))


# Complete dynamics parameters
A = two_triangles_graph_adjacency_matrix()
K = np.diag(np.sum(A, axis=1))
W = np.diag(np.array([0.2, 0.2, 0.2, -0.2, -0.2, -0.2]))
theta0 = np.linspace(0, 2*np.pi, len(A[0]))

# Reduced dynamics parameters
vapW = np.linalg.eig(W)[0]
vapK = np.linalg.eig(K)[0]
vapvep = np.linalg.eig(A)
C_W = np.array([[1/np.sqrt(3), 0],
                [0, 1/np.sqrt(3)]])
V_W = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 0],
                [0, 0, 0, 1/np.sqrt(3),  1/np.sqrt(3), 1/np.sqrt(3)]])
M = C_W @ V_W

sigma_array = np.linspace(0.01, 8, 500)

T_1, T_2, T_3 = "W", "None", "None"

synchro_transition_data_dictionary = \
    get_synchro_transition_kuramoto_graphs(t0, t1, dt, 5,
                                           T_1, T_2, T_3, theta0,
                                           M, sigma_array, W, A)

r_array = np.array(synchro_transition_data_dictionary["r"])
R_array = np.array(synchro_transition_data_dictionary["R"])

plt.plot(sigma_array, r_array)
plt.plot(sigma_array, R_array)
plt.show()

line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(len(A[0]))
line5 = "Dimension of the reduced dynamics : n = {}\n".format(len(M[:, 0]))
line6 = "theta0 = np.array({}, {}, {})\n".format(np.round(theta0[0], 3),
                                                 np.round(theta0[-1], 3),
                                                 len(theta0))
line7 = "sigma_array = np.array({}, {}, {})\n".format(np.round(sigma_array[0],
                                                               3),
                                                      np.round(sigma_array[-1],
                                                               3),
                                                      len(sigma_array))
line8 = "M = {}\n".format(M)
line9 = "T_1 = {}\n T_2 = {}\n T_3 = {}\n\n".format(T_1, T_2, T_3)
line10 = " reduction_data_dictionary is a dictionary that contains " \
         " the following keys: \n" \
         "  Keys                                                         \n " \
         "{ r,                      -> Global synchro observables        \n " \
         "                             of the complete dynamics          \n " \
         "  R,                      -> Global synchro observables        \n " \
         "                               of the reduced dynamics         \n " \
         "  sigma_array, M, l_normalized_weights, m, W, K, A,            \n " \
         "  reduced_W, reduced_K, reduced_A, n, N, theta0, T_1, T_2, T_3}\n " \


timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

if messagebox.askyesno("Python",
                       "Would you like to save the parameters, "
                       "the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    f = open('data/kuramoto/kuramoto_secIIID_article/'
             '{}_{}_dataset_informations_kuramoto'
             '.txt'.format(timestr, file), 'w')
    f.writelines(
        [line1, line2, line3, line4, line5, line6, line7, line8, "\n", line9])

    f.close()

    with open('data/kuramoto/kuramoto_secIIID_article/'
              '{}_{}_data_reduction_parameters_dictionary_kuramoto'
              '.json'.format(timestr, file), 'w') as outfile:
        json.dump(synchro_transition_data_dictionary, outfile)
