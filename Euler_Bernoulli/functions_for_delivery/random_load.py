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

