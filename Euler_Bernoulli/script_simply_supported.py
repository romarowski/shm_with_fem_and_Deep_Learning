from Euler_Bernoulli_simply_sup import *               

from scipy import linalg

import numpy as np

import matplotlib.pyplot as plt

import ipdb

n_elem = 30

K = stiffnes(n_elem)                           

f = nodal_load(1, n_elem)                                           
#f = np.zeros(40)
#f[20] =  1

#ipdb.set_trace()
d = linalg.solve(K, f)                                                  

S = stress_recovery(d, n_elem)





M = np.max(np.abs(S))

n_nodes = n_elem + 1

xnode = np.linspace(0, 1, n_nodes)

n_dofs_free = n_nodes * 2 - 3 

displ_w = d[1:n_dofs_free:2] #where 40 is the number of unconstrained nodes

displ_w = np.concatenate([[0], displ_w, [0]])

#ipdb.set_trace()

linear_loads = f[0:n_dofs_free:2]
#moment_loads = f[1:40:2]

L = 1
exact = lambda x: - 1/24 * x * (L**3 - 2*L * x**2 + x**3) 

exact_discrete = exact(xnode)

#simple plot for verification
plt.figure(1)
plt.plot(xnode, np.zeros(n_nodes), 'b--')
#plt.plot(xnode, np.zeros(n_nodes), '*')
line_FE, = plt.plot(xnode, displ_w*1e3, '*r', label = r'FE')
line_exact, = plt.plot(xnode, exact_discrete*1e3, 'k', label = r'Exact')
plt.legend(handles=[line_FE, line_exact])
plt.ylabel(r'Displacement: $w$ [mm]')
plt.xlabel(r'$x$ axis [m]')
plt.grid(True)
plt.show()



#stress recovery by averaging on the nodes

#stress = np.zeros(22)

#stress[0]  = S[0, 0]
#stress[-1] = S[-1, 1]
stress = S



#for i in range(0, np.size(S, 0) - 1):
   # stress[i + 1] = (S[i, 1] + S[i + 1, 0]) / 2

moment_exact = lambda x: -1/2 * (-x + x**2) 
#xsensor = xnode[14] #location of strain gauge

xstress = np.linspace(1 / 42, 1 - 1 / 42, n_elem)



plt.figure(2)
plt.plot(xnode, np.zeros(n_nodes), 'b--')
plt.plot(xstress, stress*1e3, 'r*')
plt.plot(xstress, moment_exact(xstress)*1e3, 'k')
plt.show()

strain_gauge = displ_w[14]



