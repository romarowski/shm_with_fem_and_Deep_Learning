def advance(t_sim, timestep, n_elem, BCs, M, C, K):
    #Generalized-alpha method for a FE discretized beam with mass matrix M,
    #damping matrix C, stiffness matrix K and n_elem elements. The simulation
    #lasts t_sim seconds with a set timestep.

    import numpy as np 
    import scipy.linalg as LA
    
    #import ipdb

    
    #fixed_dofs = 4 #Corresponding to fixed-fixed beam.
    #fixed_dofs = 2 #Cantilever beam
    fixed_dofs = np.size(BCs)
    dof_per_node = 2
    dofs = (n_elem + 1) * dof_per_node - fixed_dofs #Total nbr of DOFs
        
    h = timestep
    times = np.arange(0, t_sim + h, h) #Set time axis.
    nbr_timesteps = np.size(times)

    #Parameters for convergence.
    alpha_m = 0.2
    alpha_f = 0.4
    gamma = .5 - alpha_m + alpha_f
    beta = .25 * (gamma + .5 ) ** 2
    
    lhs = M * (1 - alpha_m) + h * C * (1 - alpha_f) * gamma \
        + h ** 2 * beta * (1 - alpha_f) * K 
    
    c1  = M * alpha_m + C * (1 - alpha_f) * h * (1 - gamma) \
        + K * h ** 2 * (1 - alpha_f) * (.5 - beta)
    
    c2 = C + K * h * (1 - alpha_f)

    
    F = uniform_load_randomized(n_elem, BCs, times, hold = 1) 
    #Gives a random vector of loads for a FE of n_elem elements, at each 
    #timestesp.
    
    #Initialize displacement, velocity and acceleration matrices.
    d = np.zeros([dofs, nbr_timesteps])
    v   = np.zeros([dofs, nbr_timesteps])
    a  = np.zeros([dofs, nbr_timesteps])
    
    ic = F[:, 0] - C @  v[:, 0] - K @ d[:, 0] #Initial conditions
    a[:, 0] = LA.solve(M, ic)

    for i in np.arange(0, np.size(times) - 1):
        #-1 because np.arange works like [a, b) i.e. b-1 is the last included
        #Time advancing. 
        
  
        
        load = (1 - alpha_f) * F[:, i+1] + alpha_f * F[:, i] 
        rhs = load  - c1 @ a[:, i] - c2 @ v[:, i] - K @ d[:, i] 
       
        a[:, i+1] = LA.solve(lhs, rhs)

        d[:, i+1] = d[:, i] + h * v[:, i] + h ** 2 * ((.5 - beta) \
                  * a[:, i] + beta * a[:, i+1])

        v[:, i+1] = v[:, i] + h * ((1 - gamma) * a[:, i] + gamma * a[:, i+1])

    return d, v, a

def random_loads_vector(n_elem, times):
    #Generates a time dependente random load vector for an Euler-Bernoulli beam
    #with fixed-fixed BCs.
    
    import numpy as np
    import random

    fixed_dofs = 4
    
    dof_per_node = 2
    dofs = (n_elem + 1) * dof_per_node - fixed_dofs
    
    nbr_timesteps = np.size(times)

    loads = np.zeros([dofs, nbr_timesteps])

    for time in range(nbr_timesteps): 
        for i in range (0, dofs, 2):
            if random.choice([True, False]):
                loads[i, time] =  np.sin(random.random() * time)

    return loads



def sine_load(n_elem, times):
    #Returns a tip sine-load for a cantilevered beam
    import numpy as np

    fixed_dofs = 2
    dof_per_node = 2
    dofs = (n_elem + 1) * dof_per_node - fixed_dofs

    nbr_timesteps = np.size(times)

    loads = np.zeros([dofs, nbr_timesteps])

    loads[-2, :] = np.sin(times)

    return loads

def uniform_load(load, n_elem, BCs):
    #Returns an uniformly distributed load
    import numpy as np
    #import ipbd
    
    q = load
    L = 1 / n_elem
    tot_nodes = n_elem + 1
    dof_per_nod = 2
    tot_dofs = tot_nodes * dof_per_nod
    
    
    r_elem = np.array([-q * L / 2, -q * L ** 2 / 12,\
                       -q * L / 2,  q * L ** 2 / 12])
    
   
    #nbr_timesteps = np.size(times)
                       
    f = np.zeros([tot_dofs])        
    
    #ipdb.set_break()    
    
    for i in range(n_elem):
        f[2*i:2*i+4] += r_elem

    for dof in BCs:
        f = np.delete(f, dof)


    return f 


def uniform_load_randomized(n_elem, BCs, time_axis, hold):
    import numpy as np
    import random
    import pdb
    
    L = 1 / n_elem
    tot_nodes = n_elem + 1
    dof_per_nod = 2
    tot_dofs = tot_nodes * dof_per_nod
    
    
    r_elem = lambda q: np.array([-q * L / 2, -q * L ** 2 / 12,\
                                 -q * L / 2,  q * L ** 2 / 12])
    
   
    nbr_timesteps = np.size(time_axis)
                       
    f = np.zeros([tot_dofs, nbr_timesteps])  
    #pdb.set_trace()
    for j, time in enumerate(time_axis):
        if time%hold == 0:
            load = random.gauss(0, 1)
        for i in range(n_elem):      
            f[2*i:2*i+4, j] += r_elem(load)

    for dof in BCs:
        f = np.delete(f, dof, 0)
        
        
    return f

