
 
def advance(t_sim, timestep, n_elem,  M, C, K):
    
    import numpy as np 
    import scipy.linalg as LA
    import ipdb

    
    fixed_dofs = 4
    dof_per_node = 2
    dofs = (n_elem + 1) * dof_per_node - fixed_dofs
        
    h = timestep
    times = np.arange(0, t_sim + h, h)
    nbr_timesteps = np.size(times)

    F = loads_vector(n_elem, times)
    
    displ = np.zeros([dofs, nbr_timesteps])
    vel   = np.zeros([dofs, nbr_timesteps])
    acel  = np.zeros([dofs, nbr_timesteps])

    lhs = M / h ** 2 + C / h
    un_term   = lhs - K 

    for i in np.arange(1, np.size(times) - 1):
        
        u = displ[:, i]
        
        rhs = F[:, i] - un_term @ displ[:, i] - M @ displ[:, i-1] / h ** 2 
        

        u_new = LA.solve(lhs, rhs)
        displ[:, i+1] = u_new
        

    return displ, F



def loads_vector(n_elem, times):
    import numpy as np

    #senoidal loads for clamped clamped beam
    fixed_dofs = 4
    
    dof_per_node = 2
    dofs = (n_elem + 1) * dof_per_node - fixed_dofs
    
    nbr_timesteps = np.size(times)

    #F = np.zeros([dofs, nbr_timesteps])

    # load = sin(t) at each node

    
    loads = np.array([np.sin(times), ] * dofs)

    return loads
