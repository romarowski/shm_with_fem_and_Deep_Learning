def simulation(n_elem, nod_loc_sg):    
    
    from Euler_Bernoulli import stress_recovery, random_load_generator, stiffnes 

    from scipy import linalg

    import numpy as np


    K = stiffnes(n_elem) # + stiff_2nd_order(21)

    loc_sg = nod_loc_sg * 2 - 2

    data = np.zeros([5000, 10])

    for i in range (0, 5000):
        f = random_load_generator(n_elem)
        d = linalg.solve(K, f)
        S = stress_recovery(d, n_elem)
        M = np.max(np.abs(S))
        sg = d[loc_sg]
        data[i, 0:-1] = sg
        data[i, -1] = M

    return data, K
