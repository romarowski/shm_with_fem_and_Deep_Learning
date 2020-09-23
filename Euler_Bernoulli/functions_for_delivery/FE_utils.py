def stiffness(n_elem, E, Iz, L, fixed_dofs): 

    #Assemblage of the stiffness matrix for a 1D Euler-Bernoulli beam.
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

def random_load_generator(n_elem, fixed_dofs):
    #Generates a random load vector f for a linear FE beam element.
    
    import random
    import numpy as np

    n_nod = n_elem + 1 #Number of nodes.
    dof_nod = 2 #Number of DOF per node.
    n_dofs = n_nod * dof_nod #Total DOFs.
    free_dofs = n_dofs - np.size(fixed_dofs) 
    f = np.zeros(free_dofs)

    for i in range(0, free_dofs, 2):
        if random.choice([True, False]):
            f[i] = random.random()   
    
    return f

def moment_recovery(displ, n_elem, L):
    
    #Recovers the moment distribution for an Euler-Bernoulli beam of lenght L
    #with a displacement field displ and number of elements n_el.
    #Moments are calculated at elementary mid-points where the approximate 
    #solution coincides with the exact one. This are known as super-convergent
    #points.

    import numpy as np

    L = L / n_elem #Length of each element

    B_x = lambda x: np.array([- 6 / L ** 2 + 12 * x / L ** 3, \
                              - 4 / L      +  6 * x / L ** 2, \
                                6 / L ** 2 - 12 * x / L ** 3, \
                              - 2 / L      +  6 * x / L ** 2])
    
    #B_x is the strain-displacement matrix. Provides the stress distribution
    #along a sigle beam element. Note that the moment fiel is linear.

    stress_elem = np.zeros([n_elem])

    displ = np.concatenate(([0, 0], displ, [0, 0])) #Addition of fixed DOFs.

    for elem in range(n_elem): #Stress calculation.
       stress_elem[elem] = - B_x(L / 2) @ displ[2*elem:2*elem+4] 

    return stress_elem

