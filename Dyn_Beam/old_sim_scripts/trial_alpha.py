
from functions import *
from gen_alpha_as_paper import *
import numpy as np
import matplotlib.pyplot as plt
import ipdb


n_elem = 21

K = stiffnes(n_elem)
M = mass_matrix(n_elem)

alpha = 0.01
beta  = 0.02

C = alpha * K + beta * M #Rayleigh proportional damping

t_sim = 5 
timestep = .01

n_sim = 5000

loc_sg = 14 #this should be improved! recover the displacement at dof 14 
data = np.zeros((int(np.ceil(t_sim/timestep) + 1), 2, n_sim))

for i in range(n_sim):
    displ, F = advance(t_sim, timestep, n_elem, M, C, K)
    stresses = stress_recovery(displ, n_elem)
    #ipdb.set_trace()
    max_stress_each_timestep = np.amax(stresses, axis = 0)
    displ_at_sg = displ[14, :]
    data[:, 0, i] = displ_at_sg
    data[:, 1, i] = max_stress_each_timestep

#generating text file 

with open('Simulation2.txt', 'w') as outfile:
    for simulation in np.arange(n_sim):
        outfile.write('# Simulation # {0}\n'.format(simulation))
        np.savetxt(outfile, data[:, :, simulation], fmt='%-7.2e')
        


#plotting
#L = 1 
#Le = L / n_elem
#
#xnod = np.linspace(0, L, n_elem + 1)
#
#
#displ, F = advance(t_sim, timestep, n_elem, M, C, K)
#
#
#
#for i in np.arange(0, np.size(displ, 1), 2):
#    plt.axes(xlim = (0, 1), ylim = (-2., 2))
#    plt.plot(xnod, np.concatenate(([0], displ[0::2, i] * 100, [0])))
#    plt.draw()
#    plt.pause(.0001)
#    plt.clf()
