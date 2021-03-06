# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


from scipy.integrate import ode
import numpy as np
import time as timer
from tqdm import tqdm
import networkx as nx
from numba import jit


# @jit(nopython=True)
def integrate_rk4(t0, t1, dt, dynamics, adjacency_matrix,
                  init_cond, *args):
    args = (adjacency_matrix, *args)
    f = dynamics
    tvec = np.arange(t0, t1, dt)
    sol = [init_cond]
    for i, t in enumerate(tvec[0:-1]):
        k1 = f(t, sol[i], *args)
        k2 = f(t+dt/2, sol[i] + k1/2, *args)
        k3 = f(t+dt/2, sol[i] + k2/2, *args)
        k4 = f(t+dt, sol[i] + k3, *args)
        sol.append(sol[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6)
    return sol


# @jit(nopython=True)
def integrate_dopri45(t0, t1, dt, dynamics, adjacency_matrix,
                      init_cond, *args):
    args = (adjacency_matrix, *args)
    f = dynamics
    tvec = np.arange(t0, t1, dt)
    sol = [init_cond]
    for i, t in enumerate(tvec[0:-1]):
        k1 = f(t, sol[i], *args)
        k2 = f(t + 1./5*dt, sol[i] + dt*(1./5*k1), *args)
        k3 = f(t + 3./10*dt, sol[i] + dt*(3./40*k1 + 9./40*k2), *args)
        k4 = f(t + 4./5*dt, sol[i] + dt*(44./45*k1 - 56./15*k2 + 32./9*k3),
               *args)
        k5 = f(t + 8./9*dt, sol[i] + dt*(19372./6561*k1 - 25360./2187*k2
                                         + 64448./6561*k3 - 212./729*k4),
               *args)
        k6 = f(t + dt, sol[i] + dt*(9017./3168*k1 - 355./33*k2 + 46732./5247*k3
                                    + 49./176*k4 - 5103./18656*k5), *args)
        v5 = 35./384*k1 + 500./1113*k3 \
            + 125./192*k4 - 2187./6784*k5 + 11./84*k6
        # k7 = f(t + dt, sol[i] + dt*v5, *args)
        # v4 = 5179./57600*k1 + 7571./16695*k3 + 393./640*k4 \
        #     - 92097./339200*k5 + 187./2100*k6 + 1./40*k7

        sol.append(sol[i] + dt*v5)

    return sol


def integrate_dopri45_nonautonomous(t0, t1, dt, dynamics, adjacency_matrix,
                                    init_cond, forcing, *args):
    args = (adjacency_matrix, *args)
    f = dynamics
    tvec = np.arange(t0, t1, dt)
    sol = [init_cond + forcing[0, :]]
    for i, t in enumerate(tvec[0:-1]):
        sol_forced = sol[i] + forcing[i, :]
        k1 = f(t, sol_forced, *args)
        k2 = f(t + 1./5*dt, sol_forced + dt*(1./5*k1), *args)
        k3 = f(t + 3./10*dt, sol_forced + dt*(3./40*k1 + 9./40*k2), *args)
        k4 = f(t + 4./5*dt, sol_forced + dt*(44./45*k1 - 56./15*k2 + 32./9*k3),
               *args)
        k5 = f(t + 8./9*dt, sol_forced + dt*(19372./6561*k1 - 25360./2187*k2
                                             + 64448./6561*k3 - 212./729*k4),
               *args)
        k6 = f(t + dt, sol_forced + dt*(9017./3168*k1 - 355./33*k2
                                        + 46732./5247*k3 + 49./176*k4
                                        - 5103./18656*k5), *args)
        v5 = 35./384*k1 + 500./1113*k3 \
            + 125./192*k4 - 2187./6784*k5 + 11./84*k6
        # k7 = f(t + dt, sol[i] + dt*v5, *args)
        # v4 = 5179./57600*k1 + 7571./16695*k3 + 393./640*k4 \
        #     - 92097./339200*k5 + 187./2100*k6 + 1./40*k7

        sol.append(sol_forced + dt*v5)

    return sol


def integrate_dynamics(t0, t1, dt, dynamics, adjacency_matrix,
                       integrator, init_cond, *args, print_process_time=False):
    """

    :param t0:
    :param t1:
    :param dt:
    :param dynamics:
    :param adjacency_matrix:
    :param integrator:
    :param init_cond:
    :param args:
    :param print_process_time:
    :return:
    """
    # print(args, len(args))
    r = ode(dynamics).set_integrator(integrator, max_step=dt)
    r.set_initial_value(init_cond, t0).set_f_params(adjacency_matrix, *args)
    t = [t0]
    sol = [init_cond]
    time_0 = timer.clock()
    i = 0
    for r.t in range(int(t1/dt)-1):
        if r.successful():
            # print(r.t+dt, r.integrate(r.t+dt))
            t.append(r.t + dt)
            sol.append(r.integrate(r.t + dt))
            i += 1
            print(i)
        # else:
        #     print("Integration was not successful for this step.")

    if print_process_time:
        print("Integration done. Time to process:",
              np.round((timer.clock()-time_0)/60, 5),
              "minutes", "(", np.round(timer.clock()-time, 5), " seconds)")

    return np.array(sol)


def get_complete_dynamics_transitions(t0, t1, dt, dynamics, graph_array,
                                      integrator, init_cond, *args):
    """

    :param t0:
    :param t1:
    :param dt:
    :param dynamics:
    :param graph_array:
    :param integrator:
    :param init_cond:
    :param args:
    :return:
    """
    nb_graph_parameters = len(graph_array[:, 0])
    nb_instances = len(graph_array[0, :])
    # nb_time_points = t1//dt
    r_avg_instances_list = []
    r_std_instances_list = []
    for i in tqdm(range(nb_graph_parameters)):
        r_list = []
        R_list = []
        for graph in graph_array[i, :]:
            time.sleep(3)
            adjacency_matrix = nx.to_numpy_array(graph)
            N = len(adjacency_matrix[:, 0])
            theta_sol = integrate_dynamics(t0, t1, dt, dynamics,
                                           adjacency_matrix, integrator,
                                           init_cond, *args)
            rt = np.absolute(np.sum(np.exp(1j*theta_sol), axis=1))/N
            r_avg = np.sum(rt[8*(t1//dt)//10:])/len(rt[8*(t1//dt)//10:])
            r_list.append(r_avg)

        r_avg_instances_list.append(np.sum(r_list)/nb_instances)
        r_std_instances_list.append(np.sum(R_list)/nb_instances)

    return r_avg_instances_list, r_std_instances_list
