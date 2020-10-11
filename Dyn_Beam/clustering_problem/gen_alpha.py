 
def advance(t_sim, timestep, n_elem,  M, C, K):
    #Generalized-alpha method for a FE discretized beam with mass matrix M,
    #damping matrix C, stiffness matrix K and n_elem elements. The simulation
    #lasts t_sim seconds with a set timestep.

    import numpy as np 
    import scipy.linalg as LA
    import ipdb

    
    fixed_dofs = 4 #Corresponding to fixed-fixed beam.
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

    F = random_loads_vector(n_elem, times) 
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

        rhs = F[:, i]  - c1 @ a[:, i] - c2 @ v[:, i] - K @ d[:, i] 
       
        a[:, i+1] = LA.solve(lhs, rhs)

        d[:, i+1] = d[:, i] + h * v[:, i] + h ** 2 * ((.5 - beta) \
                  * a[:, i] + beta * a[:, i+1])

        v[:, i+1] = v[:, i] + h * ((1 - gamma) * a[:, i] + gamma * a[:, i+1])

    return d, v, a

