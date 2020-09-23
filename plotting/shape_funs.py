import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,1)

n1 = 1 - 3 * x**2 +2 * x**3
n2 = x - 2 * x**2 + x**3
n3 = 3 * x**2 - 2 * x**3 
n4 = - x**2 + x**3

plt.subplot(1,2,1)
plt.plot(x, n1, x, n3)
plt.subplot(1,2,2)
plt.plot(x,n2,x,n4)

plt.show()
