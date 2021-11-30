from synch_predictions.graphs.special_graphs import \
    two_triangles_graph_adjacency_matrix
from synch_predictions.graphs.get_reduction_matrix_and_characteristics import *
from synch_predictions.plots.plots_setup import *


laplacian_experiment = 1
W = np.diag(np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2]))
graph_str = "two_triangles"
A = two_triangles_graph_adjacency_matrix()
N = len(A[0])
degree_sequence = np.sum(A, axis=1)
K = np.diag(degree_sequence)
Km12 = np.diag(degree_sequence**(-1/2))
L = K - A
# First eigvec. of L : (1,...,1)/sqrt(N), lambda_1 = 0
# Second eigvec. of L (Fiedler vector) : contains the partition, the related
# eigenvalue lambda_2 is called the algebraic connectivity by Fiedler(1973)
L_normalized = np.eye(N) - Km12@A@Km12


if laplacian_experiment:

    V_none = np.zeros((2, N))
    V_W = np.array([[1/np.sqrt(3), 0, 1/np.sqrt(3), 0, 1/np.sqrt(3), 0],
                    [0, 1/np.sqrt(3), 0, 1/np.sqrt(3), 0, 1/np.sqrt(3)]])

    vapvep = np.linalg.eig(L)
    print(vapvep)
    # --------------------------------- Dominance
    V0 = vapvep[1][:, 0]               # 1
    V1 = np.absolute(vapvep[1][:, 1])  # 6, =(1,...,1)/sqrt(N) for lambda1 = 0
    V2 = vapvep[1][:, 2]               # 5, lambda2 = algebraic connectivity
    V3 = vapvep[1][:, 3]               # 2
    V4 = vapvep[1][:, 4]               # 2
    V5 = vapvep[1][:, 5]               # 2

    V_L = normalize_rows_matrix_VV_T(np.array([V1, V2]))

    M_list = []
    snmf_frobenius_error_list = []
    onmf_frobenius_error_list = []
    onmf_ortho_error_list = []

    targets_arg_list = [(V_L, V_none, V_none), (V_L, V_W, V_none),
                        (V_W, V_L, V_none)]
    for targets_arg in tqdm(targets_arg_list):
        M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error = \
                get_reduction_matrix(*targets_arg)
        plt.matshow(M, aspect="auto")
        plt.colorbar()
        plt.show()
        M_list.append(M.tolist())
        snmf_frobenius_error_list.append(snmf_frobenius_error)
        onmf_frobenius_error_list.append(onmf_frobenius_error)
        onmf_ortho_error_list.append(onmf_ortho_error)

    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    if messagebox.askyesno("Python", "Would you like to save the data ? "):
        window = tkinter.Tk()
        window.withdraw()  # hides the window
        file = tkinter.simpledialog.askstring("File: ", "Enter your file name")
        with open(f'reduction_matrices_WKA/'
                  f'{timestr}_{file}_M_list.json', 'w') \
                as outfile:
            json.dump(M_list, outfile)
        with open(f'reduction_matrices_WKA/'
                  f'{timestr}_{file}_snmf_frobenius_error'
                  f'.json', 'w') as outfile:
            json.dump(snmf_frobenius_error_list, outfile)
        with open(f'reduction_matrices_WKA/'
                  f'{timestr}_{file}_onmf_frobenius_error'
                  f'.json', 'w') as outfile:
            json.dump(onmf_frobenius_error_list, outfile)
        with open(f'reduction_matrices_WKA/'
                  f'{timestr}_{file}_onmf_ortho_error'
                  f'.json', 'w') as outfile:
            json.dump(onmf_ortho_error_list, outfile)


else:
    lambda_L_max = np.linalg.eig(L)[0][0]

    X = W/0.2 + L / lambda_L_max

    vapvep = np.linalg.eig(X)
    # --------------------Dominance
    V0 = vapvep[1][:, 0]  # 1
    V1 = vapvep[1][:, 1]  # 3
    V2 = vapvep[1][:, 2]  # 2
    V3 = vapvep[1][:, 3]  # 4
    V4 = vapvep[1][:, 4]  # 5
    V5 = vapvep[1][:, 5]  # 6

    print(vapvep[0], "\n\n", vapvep[1])

    V_X = normalize_rows_matrix_VV_T(np.array([V4, V0, V2]))
    V_none = np.zeros(np.shape(V_X))

    M, esnmf, eonmf, eoonmf = get_reduction_matrix(
        V_X, V_none, V_none,
        number_initializations=1000, other_procedure=True)

    plt.matshow(M, aspect="auto")
    plt.colorbar()
    plt.show()

    print(M)
