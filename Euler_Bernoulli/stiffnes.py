#code for setting up stifnnes matrix of euler Bernoulli beam

import numpy as np
#import matplotlib.plot as plt

E  =  1
Iz =  1
L  =  1

k = np.array([[12 * E * Iz / L ** 3, 6 * E * Iz / L ** 2,\
              -12 * E * Iz / L ** 3, 6 * E * Iz / L ** 2],
	     [  6 * E * Iz / L ** 2, 4 * E * Iz / L,\
               -6 * E * Iz / L ** 2, 2 * E * Iz / L],
             [-12 * E * Iz / L ** 3,-6 * E * Iz / L ** 2,\
               12 * E * Iz / L ** 3,-6 * E * Iz / L ** 2],
             [  6 * E * Iz / L ** 2, 2 * E * Iz / L,\
               -6 * E * Iz / L ** 2 ,4 * E * Iz / L]])

dof_node = 2
dof_elem = 4
n_elem   = 21
tot_dofs = n_elem * dof_node + 2

k_struct = np.zeros([tot_dofs,tot_dofs])

for i in np.arange(0, tot_dofs, 3):
    k_struct[(np.arange(i+3), np.arange(i+3))] += k

print(k_struct)
