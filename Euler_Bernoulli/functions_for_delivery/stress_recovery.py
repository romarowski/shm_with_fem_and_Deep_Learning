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

