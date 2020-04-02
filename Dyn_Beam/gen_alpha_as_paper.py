

#this is a time advancing scheme for an euler-bernoulli beam

#ref: Paper by Chung and Hulbert

 
def advance(t_sim, timestep, n_elem,  M, C, K):
    
    import numpy as np 
    import scipy.linalg as LA
    import ipdb

    
    fixed_dofs = 4
    dof_per_node = 2
    dofs = (n_elem + 1) * dof_per_node - fixed_dofs
        
    h = timestep
    times = np.arange(0, t_sim + h, h)
    nbr_timesteps = np.size(times)

    alpha_m = 0.2
    alpha_f = 0.4
    gamma = .5 - alpha_m + alpha_f
    beta = .25 * (gamma + .5 ) ** 2

#    c1 = gamma * (1 - alpha_f) / (beta * h)
#    c2 = 1 - gamma * (1 - alpha_f) / beta
#    c3 = h * (1 - alpha_f) * (1 - gamma / (2 * beta))
#
#    m1 = (1 - alpha_m) / (beta * h ** 2)
#    m2 = (1 - alpha_m) / (beta * h)
#    m3 = 1 - (1 - alpha_m) / (2 * beta)
#
#    K_bar = K + c1 * C + m1 * M
#
    
    lhs = M * (1 - alpha_m) + h * C * (1 - alpha_f) * gamma \
        + h ** 2 * beta * (1 - alpha_f) * K
    
    c1  = M * alpha_m + C * (1 - alpha_f) * h * (1 - gamma) \
        + K * h ** 2 * (1 - alpha_f) * (.5 - beta)
    
    c2 = C + K * h * (1 - alpha_f)

    F = random_loads_vector(n_elem, times)
    
    d = np.zeros([dofs, nbr_timesteps])
    v   = np.zeros([dofs, nbr_timesteps])
    a  = np.zeros([dofs, nbr_timesteps])
    
    ic = F[:, 0] - C @  v[:, 0] - K @ d[:, 0] 
    #ipdb.set_trace()
    a[:, 0] = LA.solve(M, ic)

    for i in np.arange(0, np.size(times) - 1):#-1 because np.arange works like [a, b) i.e. b-1 is the last included
        
        rhs = F[:, i]  - c1 @ a[:, i] - c2 @ v[:, i] - K @ d[:, i]

          
        a[:, i+1] = LA.solve(lhs, rhs)

        d[:, i+1] = d[:, i] + h * v[:, i] + h ** 2 * ((.5 - beta) \
                  * a[:, i] + beta * a[:, i+1])

        v[:, i+1] = v[:, i] + h * ((1 - gamma) * a[:, i] + gamma * a[:, i+1])

    return d, F


def avg(x_old, x_new, alpha):
    return alpha * x_old + (1 - alpha) * x_new

def loads_vector(n_elem, times, alpha_f):
    import numpy as np

    #senoidal loads for clamped clamped beam
    fixed_dofs = 4
    
    dof_per_node = 2
    dofs = (n_elem + 1) * dof_per_node - fixed_dofs
    
    nbr_timesteps = np.size(times)

    #F = np.zeros([dofs, nbr_timesteps])

    # load = sin(t) at each node


    for i in np.arange(0, np.size(times) - 1):
        t_old = times[i]
        t_new = times[i+1]


        times[i] = avg(t_old, t_new, alpha_f)

    loads = np.array([np.sin(times), ] * dofs)
    

    #loads = np.zeros((dofs, nbr_timesteps))
    #loads[20, int(nbr_timesteps/2):int(nbr_timesteps/2 + 10)] = 1

    return loads

def random_loads_vector(n_elem, times):
    import numpy as np
    import random

    #random  loads for clamped clamped beam
    fixed_dofs = 4
    
    dof_per_node = 2
    dofs = (n_elem + 1) * dof_per_node - fixed_dofs
    
    nbr_timesteps = np.size(times)

    #F = np.zeros([dofs, nbr_timesteps])

    # load = sin(t) at each node
    loads = np.zeros([dofs, nbr_timesteps])

    for time in range(nbr_timesteps): 
        for i in range (0, dofs, 2):
            if random.choice([True, False]):
                loads[i, time] =  np.sin(random.random() * time)

    return loads
















