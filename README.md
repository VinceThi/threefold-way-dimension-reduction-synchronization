# threefold-way-dimension-reduction-synchronization
Code for the paper "[Threefold way to the dimension reduction of dynamics on networks: An application to synchronization](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043215)". The code is not cleaned up and a lot of simplification work should be done, along with unit tests. One should find all the necessary information (equations and important packages) to implement the method of the project in the paper:

```
@article{Thibeault2020,
author = {Thibeault, Vincent and St-Onge, Guillaume and Dub{\'{e}}, Louis J. and Desrosiers, Patrick},
doi = {10.1103/PhysRevResearch.2.043215},
journal = {Phys. Rev. Res.},
pages = {043215},
publisher = {American Physical Society},
title = {{Threefold way to the dimension reduction of dynamics on networks: An application to synchronization}},
url = {https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043215},
volume = {2},
year = {2020}
}
```

### About some important scripts: 

#### Dynamics

- `dynamics/dynamics.py`: contains the complete dynamics on graphs
- `dynamics/reduced_dynamics.py`: contains the reduced dynamics on graphs
- `dynamics/integrate.py`: contains functions to integrate arbitrary dynamics on graphs
- `dynamics/get_synchro_transition.py`: contains functions to generate the results of Section III D and E of the paper

#### Graphs

- `graphs/get_reduction_matrix_and_characteristics.py` notably contains the function `get_reduction_matrix` that return the reduction matrix given the eigenvector matrices of each target. It also contains the function `get_CVM_dictionary` that constructs a dictionary with the eigenvector matrices, the reduction matrices and some properties for the different combinations of targets.

#### Simulations

Code to generate the results from the functions in the folders `dynamics` and `graphs` and to generate the figures of the paper (note that many of them are assembled with Inkscape). IMPORTANT: The path to access the data in the scripts have to be modified by the user.

- Figure 6 of the paper is obtained with the script `simulations/synchro_transition_kuramoto_reduction_multiple_targets.py` and `plot_multiple_synchro_transition_phase_dynamics.py` (with `win_kur_theta = 1`).

- Figure 7 of the paper is obtained with the script `synchro_transition_kuramoto_reduction_different_n.py`.

- Figure 8 of the paper is obtained with the script `simulations/get_synchro_transitions_multiple_realizations.py` and `plot_multiple_synchro_transition_phase_dynamics.py` (with `win_kur_theta = 0`).

- Figure 14 of the paper is obtained from the script `simulations/synchro_transition_kuramoto_sakaguchi_one_star.py`.

- Figure 15 of the paper (Lorenz dynamics) is obtained from the Python script `simulations/chaotic_oscillators.py`.

- `simulations/synchro_transition_kuramoto_reduction_combination_WKA.py` is to answer to the question 1 of the reviewer.

- `simulations/synchro_transition_kuramoto_reduction_different_n.py` is to answer to the
question 2 of the reviewer. See also `get_M_for_each_n.py` in `graphs/two_triangles`.

Write at vincent.thibeault.1@ulaval.ca for more information.
