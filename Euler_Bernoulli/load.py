def nodal_load(load, n_elem):
    import numpy as np
    
    q = load
    L = 1 / n_elem

    r_elem = np.array([-q * L / 2, -q * L ** 2 / 12, -q * L / 2, q * L ** 2 / 12]
    
    for i in range(n_elem):
        

