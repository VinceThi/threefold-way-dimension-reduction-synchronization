from dynamics.get_reduction_errors import *
import numpy as np
import json
import tkinter.simpledialog
from tkinter import messagebox
import time

# Time parameters
t0, t1, dt = 0, 500, 0.05   # 1000, 0.05
time_list = np.linspace(t0, t1, int(t1 / dt))

q = 2
sizes = [2, 2, 2]
N = sum(sizes)
n1 = sizes[0]
n2 = sizes[1]
n3 = sizes[2]
averaging = 8

sigma_array = np.linspace(0.1, 8, 500)
omega1_array = np.linspace(0.2, 0.2, 1)

R_dictionary = get_data_kuramoto_two_triangles_3D(sigma_array, omega1_array,
                                                  averaging, t0, t1, dt,
                                                  plot_temporal_series=0)

line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "Sizes : [n1, n2, n3] = {}\n".format(sizes)
line6 = "adjacency_matrix_type = two triangles\n"
line7 = "No info here"  # "sigma = {}\n".format(sigma)
line8 = "theta0 = 2*np.pi*np.random.randn(N)\n"
line9 = "sigma_array = np.array({}, {}, {})\n".format(np.round(sigma_array[0],
                                                               3),
                                                      np.round(sigma_array[-1],
                                                               3),
                                                      len(sigma_array))
line10 = "omega1_array = np.array({}, {}, {})\n".format(omega1_array[0],
                                                        omega1_array[-1],
                                                        len(omega1_array))
# line11 = "omega = 2*[omega1] + 2*[3] + 2*[omega1]\n\n"
line11 = "omega = 3*[omega1] + 3*[-n1/n2*omega1]\n\n"
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
         "  ...                ... \n" \
         "  sigma_array,    sigma_array,\n" \
         "  omega1_array,   omega1_array}\n\n" \
         "  where the values [[---X---]] is an array of shape \n" \
         "  len(p_out_list) times len(omega_1_list) of the order parameter X" \
         " \n\nM -> Three target procedure T1 = A, T2 = K, T3 = W\n" \
         " M =  [[ 0.30827989  0.30827989  0.27033474  " \
         "0.06299859  0.02505344  0.02505344]\n" \
         " [-0.00966878 -0.00966878  0.15141561 " \
         " 0.34858439  0.25966878  0.25966878]      \n" \
         " [ 0.09449788  0.09449788 -0.11383545 " \
         " 0.11383545  0.40550212  0.40550212]]    " \
         " \n\n M_uni = M_W -> One target procedure T1 = W \n" \
         " M_0 = [[1/3, 1/3, 1/3, 0, 0, 0],\n" \
         " [0, 0, 0, 1/2, 1/2, 0],\n" \
         " [0, 0, 0, 0, 0, 1]]) = M_uni = M_W"

# " M =  np.array([[ 0.33018754,  0.33018754,  0.33018754,\n
#   0.01116455,  0.01116455, -0.01289171], " \
# "  [ 0.17067604,  0.17067604,  0.17067604,  0.17327057,
#  0.17327057, 0.14143073],                " \
# "  [-0.00086358, -0.00086358, -0.00086358,  0.31556488,
#  0.31556488, 0.37146098]])               " \
# " \n\nM_uni = M_K -> One target procedure T1 = K \n" \
# " M_0 = np.array([[1/2, 1/2, 0, 0, 0, 0], [0, 0, 1/2, 1/2, 0, 0], " \
# " [0, 0, 0, 0, 1/2, 1/2]]) = M_uni = M_K"

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

if messagebox.askyesno("Python",
                       "Would you like to save the parameters, "
                       "the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    f = open('data/kuramoto/errors_kuramoto/{}_{}_dataset_informations'
             '_for_error_kuramoto_3D.txt'.format(timestr, file), 'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, line8,
         line9, "\n", line10, line11, line12])

    f.close()

    with open('data/kuramoto/errors_kuramoto/{}_{}_data_parameters_dictionary'
              '_for_error_kuramoto_3D.json'.format(timestr, file), 'w'
              ) as outfile:
        json.dump(R_dictionary, outfile)
