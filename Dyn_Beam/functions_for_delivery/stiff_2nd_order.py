def stiff_2nd_order(n_elem, T, L, fixed_dofs):
    #Assembly of 2nd order pre-tension matrix for an Euler-Bernoulli beam
    #of elements n_elem, pre-tension T, length L and BCs fixed_dofs.
    
    import numpy as np
    
    Le = L / n_elem
    #local stiffness matrix from 2nd order term
    k = np.array([[6 / 5 * T / Le,  T / 10,\
                  -6 / 5 * T / Le,  T / 10],
                 [ T / 10, 2 * Le * T / 15,\
                  -T / 10,    -Le * T / 30 ],
                 [-6 / 5 * T / Le, -T / 10,\
                   6 / 5 * T / Le, -T / 10],
                 [ T / 10,    -Le * T / 30 ,\
                  -T / 10, 2 * Le * T / 15]])

    dof_node = 2
    dof_elem = 4
    tot_dofs = n_elem * dof_node + 2

    k_struct = np.zeros([tot_dofs,tot_dofs])

    for i in range(n_elem):
        k_struct[2*i:2*i+4, 2*i:2*i+4] += k

    for dof in fixed_dofs:
        for i in [0, 1]: 
            k_struct = np.delete(k_struct, dof, axis = i)

    return  k_struct
