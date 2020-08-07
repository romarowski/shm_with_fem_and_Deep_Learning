import matplotlib.pyplot as plt 
import numpy as np
import matplotlib as mpl
r2 = [0.688, 0.74, 0.779, 0.808, 0.829, 0.824, 0.805, 0.771, 0.721]
r2 = np.array(r2)

x = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])

plt.stem(x, r2)
plt.yticks([0,1])
plt.xlabel(r'Location of Strain-Gauge')
plt.ylabel(r'$R^2$ score')
mpl.rcParams['axes.grid'] = False
plt.show()
