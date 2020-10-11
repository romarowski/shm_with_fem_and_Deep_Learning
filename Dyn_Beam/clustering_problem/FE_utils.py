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

def mass_matrix(n_elem, rho, A, L, fixed_dofs):
    #Assembly of consistent mass matrix of an Euler-Bernoulli beam with density
    #rho, cross-sectional area A, lenght L and BCs fixed_dofs.

    import numpy as np

    Le = L / n_elem
    mass  = rho * A * Le #kg
    
    #Local mass matrix
    m = mass / 420 * np.array([[156, 22 * Le, 54, -13 * Le], 
                               [22 * Le, 4 * Le ** 2, 13 * Le, -3 * Le ** 2], 
                               [54, 13  * Le, 156, -22 * Le],
                               [-13 * Le, -3 * Le ** 2, -22 * Le, 4 * Le ** 2]])
    dof_node = 2
    dof_elem = 4
    tot_dofs = n_elem * dof_node + 2

    m_struct = np.zeros([tot_dofs,tot_dofs])

    for i in range(n_elem):
        m_struct[2*i:2*i+4, 2*i:2*i+4] += m #Assembly.

    for dof in fixed_dofs:
        for i in [0, 1]: 
            m_struct = np.delete(m_struct, dof, axis = i)
   
    return  m_struct


def moment_recovery(displ, n_elem, L):

    import numpy as np

    L = L / n_elem

    B_x = lambda x: np.array([- 6 / L ** 2 + 12 * x / L ** 3, \
                              - 4 / L      +  6 * x / L ** 2, \
                                6 / L ** 2 - 12 * x / L ** 3, \
                              - 2 / L      +  6 * x / L ** 2])
    
    timesteps = np.arange(0, np.size(displ, 1))
    stress_elem = np.zeros([n_elem, np.size(timesteps)])

    for time in timesteps:
        displ_t = np.concatenate(([0, 0], displ[:, time])) #Cantilever!
        for elem in range(n_elem):
           stress_elem[elem, time] = - B_x(L / 2) @ displ_t[2*elem:2*elem+4]

    return stress_elem


def node(x, n_elem):
    #Returns the node for a given x in [0, 1]
    
    n_nod = n_elem + 1
    
    x_axis = np.linspace(0, 1, n_nod)
    
    dof_per_node = 
    
    
    

