from functions import *
from finite_difference import *
import numpy as np
import matplotlib.pyplot as plt
import ipdb


n_elem = 21

K = stiffnes(n_elem)
M = mass_matrix(n_elem)

alpha = 0.1
beta  = 0.2

C = alpha * K + beta * M #Rayleigh proportional damping

t_sim = 2 * np.pi
timestep = .01

n_sim = 5000

loc_sg = 14 #this should be improved! recover the displacement at dof 14 
data = np.zeros((int(np.ceil(t_sim/timestep) + 1), 2, n_sim))

#for i in range(n_sim):
#    displ, F = advance(t_sim, timestep, n_elem, M, C, K)
#    stresses = stress_recovery(displ, n_elem)
#    #ipdb.set_trace()
#    max_stress_each_timestep = np.amax(stresses, axis = 0)
#    displ_at_sg = displ[14, :]
#    data[:, 0, i] = displ_at_sg
#    data[:, 1, i] = max_stress_each_timestep
#


#plotting
L = 1 
Le = L / n_elem

xnod = np.linspace(0, L, n_elem + 1)


displ_FD, F = advance(t_sim, timestep, n_elem, M, C, K)

#plotting

for i in np.arange(0, np.size(displ_FD, 1), 2):
    plt.axes(xlim = (0, 1), ylim = (-2., 2))
    plt.plot(xnod, np.concatenate(([0], displ_FD[0::2, i] * 1e3, [0])))
    plt.draw()
    plt.pause(.0001)
    plt.clf()
