import numpy as np
import matplotlib.pyplot as plt 

ax = plt.axes()                                                         

sigmoid = lambda z: np.exp(z)/(np.exp(z) + 1)

ax.plot(np.arange(-6,6,.2),sigmoid(np.arange(-6,6,.2)),'g')                                          

ax.set_ylabel(r'$\sigma(z)= \frac{e^z}{e^z+1}$')                                   
ax.set_xlabel(r'$z$')

#ax.xaxis.set_major_locator(plt.MultipleLocator(2))                         

#ax.yaxis.set_major_locator(plt.MultipleLocator(2))

#ax.spines['right'].set_position('center')
ax.set_yticks([0, .5, 1])
ax.set_xticks([0])

ax.set_xlim([-4,4])
ax.set_ylim([0,1])


plt.show()                                                             


