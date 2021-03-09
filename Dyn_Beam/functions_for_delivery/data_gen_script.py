from FE_utils import stiffness, mass_matrix, moment_recovery
from TA_utils import advance 
import numpy as np


n_elem = 20
rho = A = L = Iz = E = 1
fixed_dofs = [1, 0]  #Cantilever
#fixed_dofs = [0, 1, -2, -1] # Fixed-fixed


K = stiffness(n_elem, E, Iz, L, fixed_dofs)
M = mass_matrix(n_elem, rho, A, L, fixed_dofs)

alpha = 0.01
beta  = 0.02
C = alpha * K + beta * M #Rayleigh proportional damping

t_sim = 360 
timestep = .01

n_sensors = 9
data = np.zeros((int(np.ceil(t_sim/timestep) + 1), n_sensors + 1))

displ, vel, acel = advance(t_sim, timestep, n_elem, M, C, K) 
#Apply gen-alpha method.
moments = moment_recovery(displ, n_elem, L) #Moment calculation. 


node_loc = np.array([1, 3, 5, 7, 10, 12, 14, 16, 18])
#Equaly spaced sensors. 
loc_sg = np.arange(2, (n_elem - 2)*2, 4) #Returns tranverse displacement for equally spaced nodes only at
#[0.1 .2 .3 .4 .5 .6 .7 .8 .9]

max_moment_each_timestep = np.amax(abs(moments), axis = 0) 

displ_at_sensor = displ[loc_sg, :]
data[:, 0:-1] = displ_at_sensor.transpose()
data[:, -1] = max_moment_each_timestep

#Generate text file 
with open('./simulations/sine_simul2.txt', 'w') as outfile:
    np.savetxt(outfile, data, fmt='%-7.2e')
        
