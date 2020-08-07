def uniform_load(load, n_elem, L, fixed_dofs):
    #Assembles the nodal equivalent force vector of a uniformly distributed 
    #linear load load in [N/m], for a uniform FE discretization of n_elem,
    #of an Euler-Bernoulli beam of lenght L, with boundary conditions 
    #fixed_dofs.
   
    import numpy as np
    
    q = load
    L = L / n_elem #Element length, uniform distribution.
    dof_node = 2 #Number of DOFs per node.
    tot_dofs = n_elem * dof_node + dof_node
    
    
    r_elem = np.array([-q * L / 2, -q * L ** 2 / 12,\
                       -q * L / 2,  q * L ** 2 / 12])
    #Statically equivalent vector of forces for a 2 node beam element.
   
                       
    f = np.zeros([tot_dofs])                   
                       
    for i in range(n_elem):
        f[2*i:2*i+4] += r_elem #Assembly to structure size.

    for dof in fixed_dofs:
        f = np.delete(f, dof) #Removal of fixed DOFs.

    return f 

