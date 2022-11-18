# Predictions for the Kuramoto dynamics on a star graphs when the core has one
# natural frequency and the periphery nodes have all the same natural frequency

from dynamics.get_synchro_transition import *
import numpy as np
import json
import time

# Time parameters
t0, t1, dt = 0, 100, 0.001  # 1000, 0.05 is normally a good choice
t0_red, t1_red, dt_red = 0, 100, 0.001


# Structural parameter of the star
q = 2
Nf = 10
sizes = [1, Nf]
N = sum(sizes)
n1 = sizes[0]
n2 = sizes[1]
averaging = 9

sigma_array = np.array([1])  # np.linspace(0, 5, 500)

alpha = -0.2*np.pi    # 0.3*np.pi  # 0.7
file = f"alpha_m0_2pi"

# Important
# we make N times sigma in
# dynamics/get_reduction_errors -> get_data_kuramoto_one_star !!!
# This is only to remove the division by N
# in the definition of the complete and reduced dynamics
R_dictionary = get_data_kuramoto_sakaguchi_one_star(sigma_array, N, alpha,
                                                    t0, t1, dt, t0_red,
                                                    t1_red, dt_red,
                                                    averaging,
                                                    plot_temporal_series=1)

line1 = "t0 = {}\n".format(t0)
line2 = "t1 = {}\n".format(t1)
line3 = "deltat = {}\n".format(dt)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "Sizes : [n1, n2] = {}\n".format(sizes)
line6 = "adjacency_matrix_type = star graph\n"
line7 = "theta0 and W0 = see function get_data_kuramoto_sakaguchi_one_star\n"
line8 = "sigma_array = np.array({}, {}, {})\n".format(np.round(sigma_array[0],
                                                               3),
                                                      np.round(sigma_array[-1],
                                                               3),
                                                      len(sigma_array))
line9 = "alpha = {}\n".format(alpha)
line10 = "R_dictionary is a dictionary of the form \n\n " \
         "Keys (str)      Values         \n " \
         "{ r,             [[--- r ---]], \n" \
         "  r1,            [[--- r1---]], \n" \
         "  r2,            [[--- r2---]], \n" \
         "  R,             [[--- R ---]], \n" \
         "  ...                ... \n" \
         "  sigma_array,    sigma_array," \
         "  omega_array,    omega_array = [omega1(core), omega2(periph)] \n" \
         "  alpha,          alpha \n" \
         "    N ,             N   \n" \
         "  where the values [[---X---]] is an array of shape \n" \
         "  nb_initial_condition times len(sigma_array)of the order" \
         " parameter X\n " \
         "  R is the spectral observable (obtained with M_A: Z = M_A z)"

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

if messagebox.askyesno("Python",
                       "Would you like to save the parameters and "
                       "the data ?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    f = open('data/kuramoto_sakaguchi/{}_{}_dataset_informations'
             '_kuramoto_sakaguchi_star_2D.txt'.format(timestr, file),
             'w')
    f.writelines(
        [line1, line2, line3, "\n", line4, line5, line6, line7, line8,
         line9, "\n", line10])

    f.close()

    with open('data/kuramoto_sakaguchi/'
              '{}_{}_data_R_dictionary_kuramoto_sakaguchi_star_2D'
              '.json'.format(timestr, file),
              'w') as outfile:
        json.dump(R_dictionary, outfile)
