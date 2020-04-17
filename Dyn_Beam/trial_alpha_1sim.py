from functions import *
from gen_alpha_as_paper import *
import numpy as np
#import matplotlib.pyplot as plt
import ipdb


n_elem = 21

K = stiffnes(n_elem)
M = mass_matrix(n_elem)

alpha = 0.01
beta  = 0.02

C = alpha * K + beta * M #Rayleigh proportional damping

t_sim = 3600 
timestep = .01


loc_s = 24 #this should be improved! recover the displacement at dof 24 
data = np.zeros((int(np.ceil(t_sim/timestep) + 1), 2))


displ, vel, acel = advance(t_sim, timestep, n_elem, M, C, K)
stresses = stress_recovery(displ, n_elem)

max_stress_each_timestep = np.amax(stresses, axis = 0)
displ_at_sensor = displ[loc_s, :]
data[:, 0] = displ_at_sensor
data[:, 1] = max_stress_each_timestep

#generating text file 

with open('./simulations/Simulation5.txt', 'w') as outfile:
    np.savetxt(outfile, data, fmt='%-7.2e')
        


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
