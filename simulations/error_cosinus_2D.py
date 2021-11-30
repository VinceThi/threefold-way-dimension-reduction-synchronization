from synch_predictions.dynamics.get_reduction_errors import *
import numpy as np
import json
import tkinter.simpledialog
from tkinter import messagebox
import time

# Time parameters
t0, t1, dt = 0, 100, 0.05   # 1000, 0.05
time_list = np.linspace(t0, t1, int(t1 / dt))

# Structural parameter of the SBM
q = 2
sizes = [150, 100]
p_in = 0
mean_SBM = False
N = sum(sizes)
n1 = sizes[0]
n2 = sizes[1]
f = n1/n2

# Dynamical parameters
sigma = 1

# p_out_array = np.linspace(1/N, 1, 10)
omega1_array = np.linspace(0, 1, 10)

# R_dictionary = get_uniform_data_cosinus_2D(p_out_array, omega1_array,
#                                            sizes, t0, t1, dt, sigma,
#                                            plot_temporal_series=0,
#                                            plot_temporal_series_2=0)

R_dictionary = get_uniform_data_cosinus_2D_2(omega1_array, t0, t1, dt, sigma,
                                             plot_temporal_series=1,
                                             plot_temporal_series_2=0)


line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "Sizes : [n1, n2] = {}\n".format(sizes)
line6 = "adjacency_matrix_type = connected triangles\n"
line7 = "sigma = {}\n".format(sigma)
line8 = "theta0 = 2*np.pi*np.random.randn(N)\n"
# line9 ="p_out_array = np.array({}, {}, {})\n".format(np.round(p_out_array[0],
#                                                                3),
#                                                     np.round(p_out_array[-1],
#                                                                3),
#                                                       len(p_out_array))
line10 = "omega1_array = np.array({}, {}, {})\n".format(omega1_array[0],
                                                        omega1_array[-1],
                                                        len(omega1_array))
line11 = "omega2 = -n1/n2*omega1\n\n"
line12 = "R_dictionary is a dictionary of the form \n\n " \
         " Keys (str)      Values         \n " \
         "{ r,             [[--- r ---]], \n" \
         "  r1,            [[--- r1---]], \n" \
         "  r2,            [[--- r2---]], \n" \
         "  r_uni,         [[--- r_uni ---]], \n" \
         "  r1_uni,        [[--- r1_uni ---]], \n" \
         "  r2_uni,        [[--- r2_uni ---]], \n" \
         "  ...                ...   \n" \
         "  R_none,        [[--- R_none ---]], \n" \
         "  ...                ... \n" \
         "  R_all,         [[--- R_all ---]], \n " \
         "  ...                ...     \n" \
         "  R_hatredA,     [[--- R_hatredA ---]],  \n" \
         "   ...                ...    \n " \
         "  R_hatLambda,   [[--- R_hatLambda ---]],   \n" \
         "  ...                ...  \n" \
         "  R_uni,         [[--- R_uni ---]], \n" \
         "  ...                ...\n" \
         "  p_out_array,    p_out_array,\n" \
         "  omega1_array,   omega1_array}\n\n" \
         "  where the values [[---X---]] is an array of shape \n" \
         "  len(p_out_list) times len(omega_1_list) of the order parameter X."


timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

if messagebox.askyesno("Python",
                       "Would you like to save the parameters, "
                       "the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    f = open('data/cosinus/errors_cosinus/{}_{}_dataset_informations'
             '_for_error_cosinus_2D.txt'.format(timestr, file), 'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, line8,
         "\n", line10, line11, line12])  # line9,

    f.close()

    with open('data/cosinus/errors_cosinus/{}_{}_data_parameters_dictionary'
              '_for_error_cosinus_2D.json'.format(timestr, file), 'w'
              ) as outfile:
        json.dump(R_dictionary, outfile)

    # with open('data/cosinus/errors_cosinus/{}_{}_parameters_for'
    #           '_error_cosinus_2D.json'.format(timestr, file), 'w'
    #           ) as outfile:
    #     json.dump(parameters_matrix.tolist(), outfile)

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
#     get_random_data_cosinus_2D(sizes, nb_data, omega_min, omega_max,
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
#     f = open('data/cosinus/errors_cosinus/{}_{}_dataset_informations'
#              '_for_error_cosinus_2D.txt'.format(timestr, file), 'w')
#     f.writelines(
#         [line1, line2, line3, "\n", line4, line5, line6, line7, line8,
#          line9, "\n", line10, line11, line12])
#
#     f.close()
#
#     with open('data/cosinus/errors_cosinus/{}_{}_dataset_for'
#               '_error_cosinus_2D.json'.format(timestr, file), 'w'
#               ) as outfile:
#         json.dump(R_big_matrix.tolist(), outfile)
#
#     with open('data/cosinus/errors_cosinus/{}_{}_parameters_for'
#               '_error_cosinus_2D.json'.format(timestr, file), 'w'
#               ) as outfile:
#         json.dump(parameters_matrix.tolist(), outfile)
#
