def stiffness(n_elem, E, Iz, L, fixed_dofs): 

    #Assembly of the stiffness matrix for a 1D Euler-Bernoulli beam.
    #Is a uniformly distributed Finite Element discretization of a beam with 
    #n_elem elements, length L, Young Modulus E, Flexural Inertia Iz and 
    #boundary conditions fixed_dofs. 

    import numpy as np

    L = L / n_elem #Unioform discretization, size of each element is total 
                   # length divided by number of elements.
    
    
    k = np.array([[12 * E * Iz / L ** 3, 6 * E * Iz / L ** 2,\
                  -12 * E * Iz / L ** 3, 6 * E * Iz / L ** 2],
                 [  6 * E * Iz / L ** 2, 4 * E * Iz / L,\
                   -6 * E * Iz / L ** 2, 2 * E * Iz / L],
                 [-12 * E * Iz / L ** 3,-6 * E * Iz / L ** 2,\
                   12 * E * Iz / L ** 3,-6 * E * Iz / L ** 2],
                 [  6 * E * Iz / L ** 2, 2 * E * Iz / L,\
                   -6 * E * Iz / L ** 2 ,4 * E * Iz / L]])

    dof_node = 2 #Number of DOFs per node.

    tot_dofs = n_elem * dof_node + dof_node #Total number of DOFs. 
    
    k_struct = np.zeros([tot_dofs,tot_dofs])

    for i in range(n_elem):
        k_struct[2*i:2*i+4, 2*i:2*i+4] += k #Assembly to structure size.

    for dof in fixed_dofs:
        for i in [0, 1]: #Removal of fixed DOFs.
            k_struct = np.delete(k_struct, dof, axis = i)

    return  k_struct

