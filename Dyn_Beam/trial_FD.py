from functions import *
from finite_difference import *
import numpy as np
import matplotlib.pyplot as plt
import ipdb


n_elem = 21

K = stiffnes(n_elem)
M = mass_matrix(n_elem)

alpha = .02
beta  = .01

C = alpha * K + beta * M #Rayleigh proportional damping

t_sim = 2 * np.pi
timestep = .01

displ, F = advance(t_sim, timestep, n_elem, M, C, K)

#ipdb.set_trace()

#plotting
L = 1 
Le = L / n_elem

xnod = np.linspace(0, L, n_elem + 1)

for i in np.arange(0, np.size(displ, 1), 2):
    plt.axes(xlim = (0, 1), ylim = (-2., 2))
    plt.plot(xnod, np.concatenate(([0], displ[0::2, i] * 1e3, [0])))
    plt.draw()
    plt.pause(.0001)
    plt.clf()
