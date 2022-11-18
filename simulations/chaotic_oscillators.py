from dynamics.integrate import *
from dynamics.dynamics import *
from dynamics.reduced_dynamics import *
from plots.plots_setup import *
from graphs.special_graphs import *
from graphs.get_reduction_matrix_and_characteristics import *
from tqdm import tqdm

graph_str = "mean_SBM"        # "star", "SBM", or "mean_SBM"
dynamics_str = "lorenz"       # "rossler"-->not coded yet or "lorenz"
chaotic_oscillator = lorenz   # "rossler"-->not coded yet or "lorenz"
simulate = 1
plot_temporal_series_bool = 1


def plot_temporal_series(x_mu, y_mu, z_mu, X, Y, Z, r, r_avg, R, R_avg):
    # Plot
    plt.figure(figsize=(15, 8))

    plt.subplot(411)
    # for i in range(0, N):
    #     if i < sizes[0]:
    #         plt.plot(timelist, x[:, i], color=first_community_color,
    #                  linewidth=linewidth)
    #     else:
    #         plt.plot(timelist, x[:, i], color=second_community_color,
    #                  linewidth=linewidth)
    plt.plot(timelist, x_mu[:, 0], color=first_community_color,
             linewidth=linewidth)
    plt.plot(timelist, x_mu[:, 1], color=second_community_color,
             linewidth=linewidth)
    plt.plot(timelist, X[:, 0], color=reduced_first_community_color,
             linewidth=linewidth, linestyle="--")
    plt.plot(timelist, X[:, 1], color=reduced_second_community_color,
             linewidth=linewidth, linestyle="--")
    # ylab = plt.ylabel('$x_j$', fontsize=fontsize, labelpad=20)
    ylab = plt.ylabel('$X_{\\mu}$', fontsize=fontsize, labelpad=20)
    ylab.set_rotation(0)

    plt.subplot(412)
    # for i in range(0, N):
    #     if i < sizes[0]:
    #         plt.plot(timelist, y[:, i], color=first_community_color,
    #                  linewidth=linewidth)
    #     else:
    #         plt.plot(timelist, y[:, i], color=second_community_color,
    #                  linewidth=linewidth)
    plt.plot(timelist, y_mu[:, 0], color=first_community_color,
             linewidth=linewidth)
    plt.plot(timelist, y_mu[:, 1], color=second_community_color,
             linewidth=linewidth)
    plt.plot(timelist, Y[:, 0], color=reduced_first_community_color,
             linewidth=linewidth, linestyle="--")
    plt.plot(timelist, Y[:, 1], color=reduced_second_community_color,
             linewidth=linewidth, linestyle="--")
    # ylab2 = plt.ylabel('$y_j$', fontsize=fontsize, labelpad=20)
    ylab = plt.ylabel('$Y_{\\mu}$', fontsize=fontsize, labelpad=20)
    ylab.set_rotation(0)

    plt.subplot(413)
    # for i in range(0, N):
    #     if i < sizes[0]:
    #         plt.plot(timelist, z[:, i], color=first_community_color,
    #                  linewidth=linewidth)
    #     else:
    #         plt.plot(timelist, z[:, i], color=second_community_color,
    #                  linewidth=linewidth)
    plt.plot(timelist, z_mu[:, 0], color=first_community_color,
             linewidth=linewidth)
    plt.plot(timelist, z_mu[:, 1], color=second_community_color,
             linewidth=linewidth)
    plt.plot(timelist, Z[:, 0], color=reduced_first_community_color,
             linewidth=linewidth, linestyle="--")
    plt.plot(timelist, Z[:, 1], color=reduced_second_community_color,
             linewidth=linewidth, linestyle="--")
    # ylab3 = plt.ylabel('$z_j$', fontsize=fontsize, labelpad=20)
    ylab3 = plt.ylabel('$Z_{\\mu}$', fontsize=fontsize, labelpad=20)
    ylab3.set_rotation(0)

    # plt.subplot(514)
    # for i in range(0, N):
    #     if i < sizes[0]:
    #         plt.plot(timelist, np.cos(theta[:, i]),
    # color=first_community_color,
    #                  linewidth=linewidth)
    #     else:
    #         plt.plot(timelist, np.cos(theta[:, i]),
    # color=second_community_color,
    #                  linewidth=linewidth)
    # # plt.plot(timelist, theta_mu_array[:, 0], color=first_community_color,
    # #          linewidth=redlinewidth)
    # # plt.plot(timelist, theta_mu_array[:, 1], color=second_community_color,
    # #          linewidth=redlinewidth)
    # # plt.plot(timelist, redtheta[:, 0], color=reduced_first_community_color,
    # #          linewidth=redlinewidth, linestyle="--")
    # # plt.plot(timelist, redtheta[:, 1],
    # color=reduced_second_community_color,
    # #          linewidth=redlinewidth, linestyle="--")
    # ylab4 = plt.ylabel('$\\theta_j$', fontsize=fontsize, labelpad=20)
    # ylab4.set_rotation(0)

    plt.subplot(414)
    plt.plot(timelist, r_true, color="#373737",
             linewidth=linewidth)
    plt.plot(timelist, r_true_avg * np.ones(len(timelist)), color="#373737",
             linewidth=linewidth - 1)
    plt.plot(timelist, r, color=first_community_color,
             linewidth=linewidth - 1)
    plt.plot(timelist, r_avg * np.ones(len(timelist)),
             color=first_community_color,
             linewidth=linewidth - 1)
    plt.plot(timelist, R, color=second_community_color,
             linewidth=linewidth - 1)
    plt.plot(timelist, R_avg * np.ones(len(timelist)),
             color=second_community_color,
             linewidth=linewidth - 1)
    ylab5 = plt.ylabel('$R$', fontsize=fontsize, labelpad=20)
    ylab5.set_rotation(0)

    print(r_true_avg, r_avg, R_avg)

    plt.xlabel('$t$', fontsize=fontsize)

    # plt.ylim([9, 51])

    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(6, 6))
    # plt.plot(x[:, 0], z[:, 0],
    # color=second_community_color, linewidth=linewidth)
    # ylab = plt.ylabel('$y_j$', fontsize=fontsize, labelpad=20)
    # ylab.set_rotation(0)
    # plt.xlabel('$x_j$', fontsize=fontsize)
    # plt.tight_layout()
    # plt.show()
    # """


# Time parameters
t0, t1, dt = 0, 100, 0.01   # 1000, 0.05
timelist = np.linspace(t0, t1, int(t1 / dt))

# Structural parameter of the star
q = 2
n = 2
if graph_str == "mean_SBM":
    n1 = 30
    n2 = 20
    sizes = [n1, n2]
    pin = 0.8
    pout = 0.1
    pq = [[pin, pout],
          [pout, pin]]
    A = mean_SBM(sizes, pq, self_loop=False)
    # A = nx.to_numpy_array(nx.stochastic_block_model(sizes, pq))
else:  # graph_str == "star":
    Nf = 10
    n1 = 1
    n2 = Nf
    sizes = [n1, n2]
    A = nx.to_numpy_array(nx.star_graph(n2))

N = sum(sizes)
averaging = 9
K = np.diag(np.sum(A, axis=0))
M = np.array([np.concatenate([np.ones(n1)/n1, np.zeros(n2)]),
              np.concatenate([np.zeros(n1), np.ones(n2) / n2])])
l_weights = np.count_nonzero(M, axis=1)
l_normalized_weights = l_weights / np.sum(l_weights)

# Dynamical parameters
if dynamics_str == "lorenz":
    a, b, c = 10, 28, 8/3
else:
    a, b, c = 0.2, 0.1, 8.5

omega1 = 1  # np.diag(K)[0]
omega2 = 1  # np.diag(K)[1]
omega_array = np.array([omega1, omega2])
omega = np.array(n1*[omega1] + n2 * [omega2])
W = np.diag(omega)
Omega = (omega1 + (N-1)*omega2)/N
# omega_CM = np.array(n1*[omega1-Omega] + n2 * [omega2-Omega])

number_initial_condition = 1
sigma_array = [0.05, 0.4]  # np.linspace(0.4, 3, 100)  # np.linspace(0, 3, 50)


if simulate:
    r_true_list = []
    r_list = []
    r1_list = []
    r2_list = []
    R_list = []

    for i in tqdm(range(len(sigma_array))):
        r_true_CI = 0
        r_CI = 0
        r1_CI = 0
        r2_CI = 0
        R_CI = 0
        for j in range(number_initial_condition):
            sigma = sigma_array[i]
            # Integrate complete dynamics
            x00 = np.linspace(-10, 10, N) + 5*np.random.random(N)
            y00 = np.linspace(-10, 10, N) + 5*np.random.random(N)
            z00 = np.linspace(-10, 10, N) + 5*np.random.random(N)
            x0 = np.concatenate((x00, y00, z00))
            complete_dynamics = chaotic_oscillator
            args_complete_dynamics = (omega, sigma, a, b, c)
            complete_dynamics_sol = np.array(
                integrate_dopri45(t0, t1, dt, complete_dynamics,
                                  A, x0, *args_complete_dynamics))
            x = complete_dynamics_sol[:, 0:N]
            y = complete_dynamics_sol[:, N:2*N]
            z = complete_dynamics_sol[:, 2*N:3*N]
            u = np.sqrt(x**2 + y**2)
            if dynamics_str == "lorenz":
                z0 = 27
                u0 = 12
                theta = np.arctan((z-z0)/(u - u0))
            else:
                xdot = -y - z - sigma*(x*np.sum(A, axis=1) - A@x)
                ydot = x + a*y

                theta = np.arctan(ydot/xdot)
            r1 = np.absolute(np.sum(M[0, :] * np.exp(1j * theta), axis=1))
            r1_avg = np.mean(r1[averaging*int(t1//dt)//10:])
            r2 = np.absolute(np.sum(M[1, :] * np.exp(1j * theta), axis=1))
            r2_avg = np.mean(r2[averaging * int(t1 // dt) // 10:])
            Z_cplx = np.mean((n1*M[0, :] + n2*M[1, :])*np.exp(1j*theta),
                             axis=1)
            r_true = np.absolute(Z_cplx)
            r_true_avg = np.mean(r_true[averaging*int(t1//dt)//10:])
            x_mu = (M@x.T).T
            y_mu = (M@y.T).T
            z_mu = (M@z.T).T
            u_mu = np.sqrt(x_mu**2 + y_mu**2)
            phi_mu = np.arctan((z_mu-z0)/(u_mu - u0))
            r = np.absolute(np.sum(l_normalized_weights*np.exp(1j*phi_mu),
                                   axis=1))
            r_avg = np.mean(r[averaging*int(t1//dt)//10:])

            # Integrate reduced dynamics
            redx0 = np.ndarray.flatten(M@x00)
            redy0 = np.ndarray.flatten(M@y00)
            redz0 = np.ndarray.flatten(M@z00)
            X0 = np.concatenate((redx0, redy0, redz0))
            # print(redx0, X0)
            # X0 = np.ravel(M@np.reshape(x0, (3, N)).T, order="F")
            MWMp = get_reduced_parameter_matrix(M, W)
            MKMp = get_reduced_parameter_matrix(M, K)
            MAMp = get_reduced_parameter_matrix(M, A)
            args_reduced_lorenz = (MWMp, MKMp, sigma, Omega, a, b, c)
            reduced_lorenz_sol = np.array(
                integrate_dopri45(t0, t1, dt,
                                  reduced_lorenz,
                                  MAMp, X0,
                                  *args_reduced_lorenz))
            X = reduced_lorenz_sol[:, 0:n]
            Y = reduced_lorenz_sol[:, n:2*n]
            Z = reduced_lorenz_sol[:, 2*n:3*n]
            U = np.sqrt(X**2 + Y**2)
            Phi_mu = np.arctan((Z-z0)/(U - u0))
            R = np.absolute(np.sum(l_normalized_weights*np.exp(1j*Phi_mu),
                                   axis=1))
            R_avg = np.mean(R[averaging*int(t1//dt)//10:])

            r_true_CI += r_true_avg/number_initial_condition
            r_CI += r_avg/number_initial_condition
            r1_CI += r1_avg/number_initial_condition
            r2_CI += r2_avg/number_initial_condition
            R_CI += R_avg/number_initial_condition

            if plot_temporal_series_bool:

                # plot_temporal_series(x_mu, y_mu, z_mu, X, Y, Z,
                #                      r, r_avg, R, R_avg)

                # plt.plot(timelist, u_mu[:, 0],
                #          color=first_community_color,
                #          linewidth=linewidth, linestyle="--")

                plt.figure(figsize=(3, 3))
                ax = plt.subplot(111)
                plt.title(f"$\\sigma =$ {sigma}", fontsize=fontsize)
                plt.plot(x_mu[:, 0], z_mu[:, 0], color=first_community_color,
                         linewidth=linewidth-1.65)
                ylab = plt.ylabel('$\\mathcal{Z}_{\\mu}$', fontsize=fontsize,
                                  labelpad=10)
                ylab.set_rotation(0)
                # plt.ylim([0, 40])
                plt.tick_params(axis='both', which='major',
                                labelsize=labelsize)
                plt.xlabel('$\\mathcal{X}_{\\mu}$', fontsize=fontsize)
                plt.tight_layout()
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                for axis in ['bottom', 'left']:
                    ax.spines[axis].set_linewidth(linewidth - 1)
                plt.show()

        r_true_list.append(r_true_CI)
        r_list.append(r_CI)
        r1_list.append(r1_CI)
        r2_list.append(r2_CI)
        R_list.append(R_CI)

    line1 = f"t0 = {t0}\n"
    line2 = f"t1 = {t1}\n"
    line3 = f"deltat = {dt} \n"
    line4 = f"Number of nodes : N = {N}, N1 = {n1}, N2 = {n2}\n"
    line5 = f"adjacency_matrix_type = {graph_str}\n"
    line6 = f"number of initial conditions = {number_initial_condition}\n"
    line7 = f"sigma_array = {sigma_array}\n"
    line8 = f"dynamical parameters: a = {a}, b = {b}, c = {c}\n"
    line9 = f"omega = {omega}\n"
    line10 = "Initial conditions: " \
             "x00 = np.linspace(-10, 10, N) + 5*np.random.random(N)\n" \
             "y00 = np.linspace(-10, 10, N) + 5*np.random.random(N)\n" \
             "z00 = np.linspace(-10, 10, N) + 5*np.random.random(N)\n"

else:
    r_str = "2020_09_23_18h40min08sec_100sig_100CI_sbm_moyen_complete_r_list"
    # r1_str = ""  à ce moment, je n'avais pas le code pour avoir r1 r2
    # r2_str = ""  à ce moment, je n'avais pas le code pour avoir r1 r2
    r_true_str = "2020_09_23_18h40min08sec" \
                 "_100sig_100CI_sbm_moyen_complete_r_true_list"
    R_str = "2020_09_23_18h40min08sec_100sig_100CI_sbm_moyen_reduced_R_list"
    with open(f'data/lorenz/{r_str}.json') as json_data:
        r_list = np.array(json.load(json_data))
    # with open(f'data/lorenz/{r1_str}.json') as json_data:
    #     r1_list = np.array(json.load(json_data))
    # with open(f'data/lorenz/{r2_str}.json') as json_data:
    #     r2_list = np.array(json.load(json_data))
    with open(f'data/lorenz/{r_true_str}.json') as json_data:
        r_true_list = np.array(json.load(json_data))
    with open(f'data/lorenz/{R_str}.json') as json_data:
        R_list = np.array(json.load(json_data))

timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

fig = plt.figure(figsize=(4, 3))
plt.plot(sigma_array, r_true_list, color=reduced_third_community_color,
         label="True global sync.")
plt.plot(sigma_array, r_list, color="#252525",
         label="Complete")
plt.plot(sigma_array, R_list, color="#969696",
         label="Reduced")
# plt.plot(sigma_array, r1_list, color=first_community_color,
#          label="$\\langle R_1 \\rangle_t$")
# plt.plot(sigma_array, r2_list, color=second_community_color,
#          label="$\\langle R_2 \\rangle_t$")
ylab = plt.ylabel("$\\langle R \\rangle_t$", fontsize=fontsize+2, labelpad=10)
ylab.set_rotation(0)
plt.xlabel("$\\sigma$", fontsize=fontsize+2)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([0.6, 1.02])
plt.xlim([0, sigma_array[-1]+0.02])
plt.yticks([0.6, 0.8, 1.0])
plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the parameters, "
                       "the data and the plot?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    fig.savefig(f"data/lorenz/"
                f"{timestr}_{file}_complete_vs_reduced_{dynamics_str}_model"
                f".pdf")
    fig.savefig(f"data/lorenz/"
                f"{timestr}_{file}_complete_vs_reduced_{dynamics_str}_model"
                f".png")

    if simulate:
        f = open(f'data/lorenz/{timestr}_{file}_parameters_{dynamics_str}.txt',
                 'w')
        f.writelines(
            [line1, line2, line3, "\n", line4, line5, line6, line7,
             "\n", line8, line9, line10])

        f.close()

        with open(f'data/{dynamics_str}/{timestr}_{file}_complete'
                  f'_{dynamics_str}'
                  f'_r_true_list.json', 'w') as outfile:
            json.dump(r_true_list, outfile)
        with open(f'data/{dynamics_str}/{timestr}_{file}_complete'
                  f'_{dynamics_str}'
                  f'_r_list.json', 'w') as outfile:
            json.dump(r_list, outfile)
        with open(f'data/{dynamics_str}/{timestr}_{file}_complete'
                  f'_{dynamics_str}'
                  f'_r1_list.json', 'w') as outfile:
            json.dump(r1_list, outfile)
        with open(f'data/{dynamics_str}/{timestr}_{file}_complete'
                  f'_{dynamics_str}'
                  f'_r2_list.json', 'w') as outfile:
            json.dump(r2_list, outfile)
        with open(f'data/{dynamics_str}/{timestr}_{file}_reduced'
                  f'_{dynamics_str}'
                  f'_R_list.json', 'w') \
                as outfile:
            json.dump(R_list, outfile)
