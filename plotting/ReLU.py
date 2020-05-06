import matplotlib.pyplot as plt 

ax = plt.axes()                                                         

ax.plot([-2, 0, 2],[0, 0, 2],'g')                                          

ax.set_ylabel(r'$g(z) = \mathrm{max} \{0,z\}$')                                   
ax.set_xlabel(r'$z$')

ax.xaxis.set_major_locator(plt.MultipleLocator(2))                         

ax.yaxis.set_major_locator(plt.MultipleLocator(2))

ax.set_xlim([-1,1])
ax.set_ylim([-.1,1])


plt.show()                                                             


