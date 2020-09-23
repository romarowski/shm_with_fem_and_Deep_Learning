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
