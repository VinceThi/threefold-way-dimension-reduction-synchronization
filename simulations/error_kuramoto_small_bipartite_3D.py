from dynamics.get_reduction_errors import *
import numpy as np
import json
import tkinter.simpledialog
from tkinter import messagebox
import time

# Read the code before running it

# Time parameters
t0, t1, dt = 0, 500, 0.05   # 1000, 0.05
time_list = np.linspace(t0, t1, int(t1 / dt))

q = 2
sizes = [1, 1, 4]
# p_in = 0
# mean_SBM = False
N = sum(sizes)
n1 = sizes[0]
n2 = sizes[1]
averaging = 5

sigma_array = np.linspace(0.1, 4, 500)
omega1_array = np.linspace(0.2, 0.2, 1)

R_dictionary = get_data_kuramoto_small_bipartite_3D(sigma_array, omega1_array,
                                                    averaging, t0, t1, dt,
                                                    plot_temporal_series=0)

line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "Sizes : [n1, n2, n3] = {}\n".format(sizes)
line6 = "adjacency_matrix_type = small SBM\n"
line7 = "No info here"  # "sigma = {}\n".format(sigma)
line8 = "theta0 = np.linspace(0, 2*np.pi*, N)\n"
line9 = "sigma_array = np.array({}, {}, {})\n".format(np.round(sigma_array[0],
                                                               3),
                                                      np.round(sigma_array[-1],
                                                               3),
                                                      len(sigma_array))
line10 = "omega1_array = np.array({}, {}, {})\n".format(omega1_array[0],
                                                        omega1_array[-1],
                                                        len(omega1_array))
# None\n"
#
#
# line11 = "omega = 2*[omega1] + 2*[3] + 2*[omega1]\n\n"
line11 = "omega = 3*[omega1] + 3*[-n1/n2*omega1]\n\n  "  # MAYBE TO CHANGE for
# np.diag(K)/10 = [0.1, 0.3, 0.2, 0.2, 0.2, 0.2]\n\n   # <--- this
line12 = "R_dictionary is a dictionary of the form \n\n " \
         "Keys (str)      Values         \n " \
         "{ r,             [[--- r ---]], \n" \
         "  r1,            [[--- r1---]], \n" \
         "  r2,            [[--- r2---]], \n" \
         "  r3,            [[--- r3---]], \n" \
         "  r_uni,         [[--- r_uni ---]], \n" \
         "  r1_uni,        [[--- r1_uni ---]], \n" \
         "  r2_uni,        [[--- r2_uni ---]], \n" \
         "  r3_uni,        [[--- r3_uni ---]], \n" \
         "  R,             [[--- R ---]], \n" \
         "  ...                ... \n" \
         "  R_uni,         [[--- R_uni ---]], \n" \
         "  ...                ...\n" \
         "  sigma_array,    sigma_array,\n" \
         "  omega1_array,   omega1_array}\n\n" \
         "  where the values [[---X---]] is an array of shape \n" \
         "  len(p_out_list) times len(omega_1_list) " \
         "  of the order parameter X.\n " \
         "  R is the spectral observable (obtained with M_A: Z = M_A z) \n" \
         "  R_uni is the degree observable" \
         " (obtained with M_0 = M_T = M_K: Z = M_K z)"

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

if messagebox.askyesno("Python",
                       "Would you like to save the parameters, "
                       "the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    f = open('data/kuramoto/errors_kuramoto/{}_{}_dataset_informations'
             '_for_error_kuramoto_small_bipartite_3D.txt'.format(timestr,
                                                                 file), 'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, line8,
         line9, "\n", line10, line11, line12])

    f.close()

    with open('data/kuramoto/errors_kuramoto/{}_{}_data_parameters_dictionary'
              '_for_error_kuramoto_small_bipartite_3D.json'.format(timestr,
                                                                   file), 'w'
              ) as outfile:
        json.dump(R_dictionary, outfile)
