
 #this is a time advancing scheme for an euler-bernoulli beam

 #ref: https://fenicsproject.org/docs/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html

 
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
    gamma = .5 + alpha_m - alpha_f
    beta = .25 * (gamma + .5 ) ** 2

    c1 = gamma * (1 - alpha_f) / (beta * h)
    c2 = 1 - gamma * (1 - alpha_f) / beta
    c3 = h * (1 - alpha_f) * (1 - gamma / (2 * beta))

    m1 = (1 - alpha_m) / (beta * h ** 2)
    m2 = (1 - alpha_m) / (beta * h)
    m3 = 1 - (1 - alpha_m) / (2 * beta)

    K_bar = K + c1 * C + m1 * M

    F = loads_vector(n_elem, times, alpha_f)
    
    displ = np.zeros([dofs, nbr_timesteps])
    vel   = np.zeros([dofs, nbr_timesteps])
    acel  = np.zeros([dofs, nbr_timesteps])
 
    for i in np.arange(0, np.size(times) - 1):
        
        u = displ[:, i]
        v =   vel[:, i]
        a =  acel[:, i]


        lhs = F[:, i] - alpha_f * K @ u - C @ ( c1 * u + c2 * v + c3 * a) - \
                M @ (m1 * u + m2 * v + m3 * a)
        

        u_new = LA.solve(K_bar, lhs)
        displ[:, i+1] = u_new
        
        a_new = update_a(u_new, u, v, a, beta, h)
        acel[:, i+1] = a_new

        v_new = update_v(a_new, u, v, a, gamma, h)
        vel[:, i+1] = v_new
        #ipdb.set_trace()

    return displ, vel, acel, F

def update_a(u, u_old, v_old, a_old, beta, h):
    return (u-u_old-h*v_old)/beta/h**2-(1-2*beta)/2/beta*a_old

def update_v(a, u_old, v_old, a_old, gamma, h):
    return v_old + h*((1-gamma)*a_old + gamma*a)

def avg(x_old, x_new, alpha):
    return alpha * x_old + (1 - alpha) * x_new

def loads_vector(n_elem, times, alpha_f):
    import numpy as np

    #senoidal loads for clamped clamped beam
    fixed_dofs = 4
    
    dof_per_node = 2
    dofs = (n_elem + 1) * dof_per_node - fixed_dofs
    
    nbr_timesteps = np.size(times)

    F = np.zeros([dofs, nbr_timesteps])

    # load = sin(t) at each node

    
    loads = np.array([np.sin(times), ] * dofs)
    
    

    for i in np.arange(0, np.size(times) - 1):
        f_old = loads[:, i]
        f_new = loads[:, i+1]


        F[:, i+1] = avg(f_old, f_new, alpha_f)


    return F

















