from FE_utils import stiffness, mass_matrix, moment_recovery, node_mapping
from TA_utils import advance 
import numpy as np
import pdb


n_elem = 20
rho = A = L = E = Iz = 1

fixed_dofs = [1, 0] #Cantilever


K = stiffness(n_elem, E, Iz, L, fixed_dofs)
M = mass_matrix(n_elem, rho, A, L, fixed_dofs)

alpha = 0.01
beta  = 0.02
C = alpha * K + beta * M #Rayleigh proportional damping

t_sim = 360 
timestep = .01

n_sensors = 10
data = np.zeros((int(np.ceil(t_sim/timestep) + 1), n_sensors + 1))

BCs = fixed_dofs

displ, vel, acel = advance(t_sim, timestep, n_elem, BCs, M, C, K) #Apply gen-alpha method.

displ = np.pad(displ, ((2,0), (0,0)), 'constant') #post process add zeros for fixed dofs
               
#REMEMBER TOCHANGE BCs for moment recovery
moments = moment_recovery(displ, n_elem, L) #Moment calculation. 


#Ten equally spaced sensors, cantilever at x=0
sensors_location = np.linspace(.1, 1, 10)

dof_measured = np.zeros(sensors_location.size)

for i, sensor_x in enumerate(sensors_location):

    dof_measured[i] = node_mapping(sensor_x , n_elem)[0][0]
    
max_moment_each_timestep = np.amax(abs(moments), axis = 0) 

#pdb.set_trace()
displ_at_sensor = displ[dof_measured.astype(int), :]
data[:, 0:-1] = displ_at_sensor.transpose()
data[:, -1] = max_moment_each_timestep



#Generate text file 
with open('./simulations/s4_cantilever_Tip_sine_load.txt', 'w') as outfile:
    np.savetxt(outfile, data, fmt='%-7.2e')
        
