from synch_predictions.dynamics.get_reduction_errors import *
import numpy as np
import json
import tkinter.simpledialog
from tkinter import messagebox
import time

# Time parameters
t0, t1, dt = 0, 100, 0.05   # 1000, 0.05
time_list = np.linspace(t0, t1, int(t1 / dt))


# Dynamical parameters
sigma = 1
p_out_array = np.linspace(0.1, 1, 20)
omega1_array = np.linspace(0, 1, 20)
N_array = np.array([10000])  # 10, 50, 100, 500, 1000])  # ,
# np.arange(10, 10000, 200)   # 30 # len(np.arange(10, 1000, 30)) = 34


R_dictionary = get_data_kuramoto_bipartite_vs_N(p_out_array, omega1_array,
                                                N_array, t0, t1, dt,
                                                plot_temporal_series=0)

line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "adjacency_matrix_type = random SBM\n"
line5 = "sigma = {}\n".format(sigma)
line6 = "theta0 = np.arccos(1/np.arange(1, N+1))\n"
# TO CHANGE HERE IF CHANGED IN THE get_data_kuramoto_bipartite_vs_N
# 2*np.pi*np.random.randn(N)\n"
line7 = "p_out_array = np.array({}, {}, {})\n".format(np.round(p_out_array[0],
                                                               3),
                                                      np.round(p_out_array[-1],
                                                               3),
                                                      len(p_out_array))
line8 = "omega1_array = np.array({}, {}, {})\n".format(omega1_array[0],
                                                       omega1_array[-1],
                                                       len(omega1_array))
line9 = "omega2 = -n1/n2*omega1\n\n"
# line10 = "R_dictionary is a dictionary of the form\n\n" \
#     "    Keys             Values                       \n" \
#     "  { r{}.format(N),             [[--- r ---]],     \n" \
#     "    r1{}.format(N),            [[--- r1---]],     \n" \
#     "    r2{}.format(N),            [[--- r2---]],     \n" \
#     "    r_uni{}.format(N),         [[--- r_uni ---]], \n" \
#     "    r1_uni{}.format(N),        [[--- r1_uni ---]],\n" \
#     "    r2_uni{}.format(N),        [[--- r2_uni ---]],\n" \
#     "    R{}.format(N),             [[--- R ---]],     \n" \
#     "    ...                ...                        \n" \
#     "    R_uni{}.format(N),         [[--- R_uni ---]], \n" \
#     "    ...                ...                        \n" \
#     "    p_out_array,    p_out_array,                  \n" \
#     "    omega1_array,   omega1_array                  \n" \
#     "    N_array,   N_array}                           \n" \
#     " where the values [[---X---]] is an array of shape\n                " \
#     " len(sigma_list) times len(omega_1_list) of the order parameter X.\n" \
#     " R is the spectral observable (obtained with M_A: Z = M_A z)      \n" \
#     " R_uni is the frequency observable (obtained with M_0 = M_W:Z=M_W z)" \
#     "                                              \n or M_T             "
line10 = "R_dictionary is a dictionary of the form\n\n" \
    "    Keys             Values                       \n" \
         "  { r{}.format(N),             [[--- r ---]],     \n" \
         "    r1{}.format(N),            [[--- r1---]],     \n" \
         "    r2{}.format(N),            [[--- r2---]],     \n" \
         "    R{}.format(N),             [[--- R ---]],     \n" \
         "    ...                ...                        \n" \
         "    p_out_array,    p_out_array,                  \n" \
         "    omega1_array,   omega1_array                  \n" \
         "    N_array,   N_array}                           \n" \
         " where the values [[---X---]] is an array of shape\n" \
         " len(sigma_list) times len(omega_1_list) of the order" \
         " parameter X.\n" \
         " R is the spectral observable (obtained with M_A: Z = M_A z) \n" \
         " R_uni is the frequency observable" \
         " (obtained with M_0 = M_W:Z=M_W z) or M_T "

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")


if messagebox.askyesno("Python",
                       "Would you like to save the parameters, "
                       "the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    f = open('data/kuramoto/errors_kuramoto/{}_{}_dataset_informations'
             '_for_error_kuramoto_2D_vs_N.txt'.format(timestr, file), 'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, line8,
         line9, "\n", line10])

    f.close()

    with open('data/kuramoto/errors_kuramoto/{}_{}_data_parameters_dictionary'
              '_for_error_kuramoto_2D_vs_N.json'.format(timestr, file), 'w'
              ) as outfile:
        json.dump(R_dictionary, outfile)

# first_community_color = "#2171b5"
# second_community_color = "#f16913"
# fontsize = 12
# inset_fontsize = 9
# fontsize_legend = 12
# labelsize = 12
# inset_labelsize = 9
# linewidth = 2
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.unicode'] = True
#
# # Time parameters
# t0, t1, dt = 0, 2000, 0.05   # 1000, 0.05
# time_list = np.linspace(t0, t1, int(t1 / dt))
#
# # Structural parameter of the SBM
# q = 2
# sizes = [300, 200]
# p_in = 0
# mean_SBM = False
# N = sum(sizes)
# n1 = sizes[0]
# n2 = sizes[1]
# f = n1/n2
#
# # Dynamical parameters
# sigma = 1
#
# omega_min = -1
# omega_max = 1
#
# nb_data = 1000
#
# R_big_matrix, parameters_matrix = \
#     get_random_data_kuramoto_2D(sizes, nb_data, omega_min, omega_max,
#                                 t0, t1, dt, sigma, plot_temporal_series=0)
#
# line1 = "t0 = {}\n".format(t0)
# line2 = "t1 = {}\n".format(t1)
# line3 = "deltat = {}\n".format(dt)
# line4 = "Number of nodes : N = {}\n".format(N)
# line5 = "Sizes : [n1, n2] = {}\n".format(sizes)
# line6 = "adjacency_matrix_type = random SBM\n"
# line7 = "nb of data (graphs, initial conditions," \
#         " pout, omega1 and omega_2 = {}\n".format(nb_data)
# line8 = "sigma = {}\n".format(sigma)
# line9 = "theta0 = 2*np.pi*np.random.randn(N)\n"
# line10 = "[omega_min, omega_max] = [{}, {}]\n".format(omega_min, omega_max)
# line11 = "p_out is randomly choosen between [1/np.sqrt(N), 1] uniformly\n\n"
# line12 = "R_big_matrix is a 21 by nb_data matrix of the form\n" \
#          "[[--- r ---],\n [--- r_uni ---],\n [--- R_none ---],\n" \
#          " [--- R_all ---],\n [--- R_hatredA ---],\n" \
#          " [--- R_hatLambda ---],\n [--- R_uni ---]] \n\n"
# line13 = "parameters_matrix is a 2 by nb_data matrix of the form\n" \
#          "[[--- p_out ---],\n [--- omega1 ---],\n [--- omega_2 ---]]"
#
# timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
#
# if messagebox.askyesno("Python",
#                        "Would you like to save the parameters, "
#                        "the data and the plot?"):
#     window = tkinter.Tk()
#     window.withdraw()  # hides the window
#     file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
#
#     f = open('data/kuramoto/errors_kuramoto/{}_{}_dataset_informations'
#              '_for_error_kuramoto_2D.txt'.format(timestr, file), 'w')
#     f.writelines(
#         [line1, line2, line3, "\n", line4, line5, line6, line7, line8,
#          line9, "\n", line10, line11, line12])
#
#     f.close()
#
#     with open('data/kuramoto/errors_kuramoto/{}_{}_dataset_for'
#               '_error_kuramoto_2D.json'.format(timestr, file), 'w'
#               ) as outfile:
#         json.dump(R_big_matrix.tolist(), outfile)
#
#     with open('data/kuramoto/errors_kuramoto/{}_{}_parameters_for'
#               '_error_kuramoto_2D.json'.format(timestr, file), 'w'
#               ) as outfile:
#         json.dump(parameters_matrix.tolist(), outfile)
#
