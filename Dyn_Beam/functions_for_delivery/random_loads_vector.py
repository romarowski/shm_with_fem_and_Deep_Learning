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





