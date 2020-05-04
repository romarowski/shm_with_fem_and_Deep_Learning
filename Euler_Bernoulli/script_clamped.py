from Euler_Bernoulli import *               

from scipy import linalg

import numpy as np

import matplotlib.pyplot as plt

K = stiffnes(21)                           

f = nodal_load(1, 21)                                           
#f = np.zeros(40)
#f[20] =  1


d = linalg.solve(K, f)                                                  
S = stress_recovery(d, 21)




M = np.max(np.abs(S))

xnode = np.linspace(0, 1, 22)

displ_w = d[0:40:2] #where 40 is the number of unconstrained nodes

displ_w = np.concatenate([[0], displ_w, [0]])

linear_loads = f[0:40:2]
#moment_loads = f[1:40:2]

#simple plot for verification
plt.figure(1)
plt.plot(xnode, np.zeros(22))
plt.plot(xnode, np.zeros(22), '*')
plt.plot(xnode, displ_w)
plt.show()

#stress recovery by averaging on the nodes

#stress = np.zeros(22)

#stress[0]  = S[0, 0]
#stress[-1] = S[-1, 1]
stress = S



#for i in range(0, np.size(S, 0) - 1):
   # stress[i + 1] = (S[i, 1] + S[i + 1, 0]) / 2


#xsensor = xnode[14] #location of strain gauge

xstress = np.linspace(1 / 42, 1 - 1 / 42, 21)

plt.figure(2)
plt.plot(xnode, np.zeros(22))
plt.plot(xnode, np.zeros(22), '*')
plt.plot(xstress, stress, '*')
plt.show()

strain_gauge = displ_w[14]



