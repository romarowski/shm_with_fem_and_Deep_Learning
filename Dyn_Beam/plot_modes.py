from ss_functions import *
from scipy import linalg as LA
import matplotlib.pyplot as plt 
import numpy as np
n_elem = 30 

K = stiffnes(n_elem)
M = mass_matrix(n_elem)

w, v = LA.eig(K, M)

w = w.real
idx = w.argsort()   
w = w[idx]
v = v[:,idx]


n_nodes = n_elem + 1

xnode = np.linspace(0, 1, n_nodes)

n_dofs_free = n_nodes * 2 - 3 

fig, axs = plt.subplots(3)

fig.text(0.5, 0.04, '$x$  axis [m]', ha='center', va='center')
fig.text(0.06, 0.5, 'Displacement: w [mm]', ha='center', va='center', rotation='vertical')
for mode in [1, 2, 3]: 

    d = v[:,mode - 1]  

    displ_w = d[1:n_dofs_free:2] #where 40 is the number of unconstrained nodes

    displ_w = displ_w[::-1]

    displ_w /= np.max(displ_w)

    displ_w = np.concatenate([[0], displ_w, [0]])

    exact = lambda x, n: np.sin(n * np.pi * x)

    exact_discrete = exact(xnode, mode)

    axs[mode-1].plot(xnode, np.zeros(n_nodes), 'b--')
    #plt.plot(xnode, np.zeros(n_nodes), '*')
    line_FE = axs[mode-1].plot(xnode, displ_w, '*r', label = 'FE')
    line_exact = axs[mode-1].plot(xnode, exact_discrete, 'k', label = 'Exact')
    #axs[mode-1].legend()
    #axs[0].ylabel('Displacement: w [mm]')
    #axs[0].xlabel('$x$ axis [m]')
    axs[mode-1].grid(True)

#box = axs[1].get_position()
#axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
#
#
#box = axs[0].get_position()
#axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
#
#
#box = axs[2].get_position()
#axs[2].set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
#Put a legend to the right of the current axis
axs[0].legend(loc='upper left')

plt.show()
