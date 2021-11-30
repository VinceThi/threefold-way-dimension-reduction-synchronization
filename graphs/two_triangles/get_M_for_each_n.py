from synch_predictions.graphs.get_reduction_matrix_and_characteristics import *
from synch_predictions.graphs.special_graphs import \
    two_triangles_graph_adjacency_matrix
from synch_predictions.plots.plots_setup import *

simulate = False


if simulate:
    # Define matrices
    A = two_triangles_graph_adjacency_matrix()
    N = len(A[0])
    graph_str = "two_triangles"
    K = np.diag(np.sum(A, axis=1))
    W = np.diag(np.array([0.1, 0.1, -0.2, -0.2, 0.1, 0.1]))

    # Eigenvector matrices of W
    VW_n1 = np.array([[0, 0, 0.5, 0.5, 0, 0]])
    VW_n2 = np.array([[0, 0, 0.5, 0.5, 0, 0],
                      [0.25, 0.25, 0, 0, 0.25, 0.25]])
    VW_n3 = np.array([[0.5, 0.5, 0, 0, 0, 0],
                      [0, 0, 0.5, 0.5, 0, 0],
                      [0, 0, 0, 0, 0.5, 0.5]])
    VW_n4 = np.array([[0.5, 0.5, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0.5, 0.5]])
    VW_n5 = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0.5, 0.5]])
    V_W_list = [VW_n1, VW_n2, VW_n3, VW_n4, VW_n5]

    # Eigenvector matrices of A
    vapvep = np.linalg.eig(A)
    V0 = np.absolute(vapvep[1][:, 0])  # || ensures strict posit. of dom. eigv.
    V1 = vapvep[1][:, 1]
    V2 = vapvep[1][:, 2]
    V3 = vapvep[1][:, 3]
    V4 = vapvep[1][:, 4]
    V5 = vapvep[1][:, 5]
    dominance_ordered_eigenvector_matrix = np.array([V0, V1, V2, V4, V5, V3])

    V_norm = normalize_rows_matrix_VV_T(dominance_ordered_eigenvector_matrix)

    print(vapvep[0], vapvep[1], "\n\n", dominance_ordered_eigenvector_matrix)

    M_n1 = normalize_rows_matrix_M1(np.array([V_norm[0]]))

    M_list = []  # [M_n1.tolist()]
    snmf_frobenius_error = []
    onmf_frobenius_error = []
    onmf_ortho_error = []
    for n in [3]:  # [1, 2, 3, 4, 5]:

        print(f"\n\n\n------------ n = {n} ------------")

        V_none = np.zeros((n, N))
        V_W = V_W_list[n-1]
        if n == 2:
            V_A = np.array([V0, V3])
        # elif n == 3:
        #     V_A = np.array([V0, V1, V3])-> I obtained bad partition with that
        else:
            V_A = V_norm[:n, ]
        M, esnmf, eonmf, eoonmf = get_reduction_matrix(
            V_A, V_W, V_none, number_initializations=5000,
            other_procedure=True)
        print(M)

        # print(np.shape(M))
        plt.matshow(M, aspect="auto")
        plt.colorbar()
        plt.yticks([])
        plt.show()

        M_list.append(M.tolist())
        snmf_frobenius_error.append(esnmf)
        onmf_frobenius_error.append(eonmf)
        onmf_ortho_error.append(eoonmf)
    M_list.append(np.eye(6).tolist())
    snmf_frobenius_error.append(0)
    onmf_frobenius_error.append(0)
    onmf_ortho_error.append(0)
    # print(M_list)

    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    fig = plt.figure(figsize=(5, 5))
    plt.plot([1, 2, 3, 4, 5, 6], snmf_frobenius_error, label="SNMF frob. err.")
    plt.plot([1, 2, 3, 4, 5, 6], onmf_frobenius_error, label="ONMF frob. err.")
    plt.plot([1, 2, 3, 4, 5, 6], onmf_ortho_error, label="ONMF ortho. err.")
    plt.xlabel("$n$", fontsize=fontsize)
    plt.ylabel("Error", fontsize=fontsize)
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.legend(loc="best", fontsize=fontsize_legend)
    plt.tight_layout()
    plt.show()
    if messagebox.askyesno("Python",
                           "Would you like to save the parameters, "
                           "the data and the plot?"):
        window = tkinter.Tk()
        window.withdraw()  # hides the window
        file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

        fig.savefig(f"reduction_matrices_n/{timestr}_{file}_errors_vs_n.pdf")
        fig.savefig(f"reduction_matrices_n/{timestr}_{file}_errors_vs_n.png")

        with open(f'reduction_matrices_n/{timestr}_{file}_M_list.json', 'w')\
                as outfile:
            json.dump(M_list, outfile)
        with open(f'reduction_matrices_n/{timestr}_{file}_snmf_frobenius_error'
                  f'.json', 'w') as outfile:
            json.dump(snmf_frobenius_error, outfile)
        with open(f'reduction_matrices_n/{timestr}_{file}_onmf_frobenius_error'
                  f'.json', 'w') as outfile:
            json.dump(onmf_frobenius_error, outfile)
        with open(f'reduction_matrices_n/{timestr}_{file}_onmf_ortho_error'
                  f'.json', 'w') as outfile:
            json.dump(onmf_ortho_error, outfile)

else:
    # M_list_path = "2020_10_03_18h20min32sec_two_targets_A_W_M_list"
    # M_list_path = "2020_10_03_18h20min32sec_two_targets_A_W_M_list_corrected"
    M_list_path = "2020_10_03_18h20min32sec_two_targets_A_W_M_list_corrected" \
                  "_n2_and_n3"
    with open(f'C:/Users/thivi/Documents/GitHub/'
              f'network-synch/synch_predictions/'
              f'graphs/two_triangles/reduction_matrices_n/{M_list_path}'
              f'.json') \
            as json_data:
        M_list = json.load(json_data)

    fig, axs = plt.subplots(2, 3, figsize=(12, 4))
    axs = axs.ravel()
    for i, M in enumerate(M_list):
        axs[i].matshow(M, aspect="auto")
        axs[i].set_title(f"$n =$ {i+1}", y=1.15)
        print(f"n = {i+1}: M = {M}\n\n")
    plt.tight_layout()
    plt.show()
