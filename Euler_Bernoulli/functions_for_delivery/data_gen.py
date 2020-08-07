def simulation(n_elem, nod_loc_sg, E, Iz, L, fixed_dofs, n_sim):    
    
    #Solves n_sim times an Euler-Bernoulli beam problem with structural 
    #properties E, Iz, L and number of elements n_elem and BCs fixed_dofs
    #The nod_loc_sg allows the user to pick from which node the transversal
    #displacement is recovered. 

    from FE_utils import moment_recovery, random_load_generator, stiffness
    from scipy import linalg
    import numpy as np

    K = stiffness(n_elem, E, Iz, L, fixed_dofs) 

    loc_sg = nod_loc_sg * 2 - 2 #Assuming sensor is located at a node. 
    #This transforms from nodal number to DOF.

    data = np.zeros([n_sim, np.size(nod_loc_sg)+1])
    #The data array will be of size [n_sim, nbr of sensor + 1]
    # +1 comes from the moment calculation

    for i in range (n_sim):
        f = random_load_generator(n_elem, fixed_dofs)
        d = linalg.solve(K, f)
        S = moment_recovery(d, n_elem, L)
        M = np.max(np.abs(S))
        sg = d[loc_sg]
        data[i, 0:-1] = sg
        data[i, -1] = M

    return data 
