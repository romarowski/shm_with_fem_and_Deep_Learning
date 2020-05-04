from plot_sdof import advance
import matplotlib.pyplot as plt
import numpy as np

t_sim = 10 
timestep = .3

times = np.arange(0, t_sim+timestep, timestep)
times_excat = np.arange(0, t_sim+timestep, timestep/10)
n_elem = 0 

C=1
K=1
M=1

ksi = C  / 2 / (K * M) ** .5
w = K/M
wd = w * (1-ksi**2)**.5
print(ksi)
d, v, a = advance(t_sim, timestep, n_elem, M, C, K)

A = ((wd**2+ksi**2*w**2)/wd**2)**.5
phi = np.arctan(wd/ksi*w)

exact_under = lambda t: A * np.exp(-ksi*w*t)*np.sin(wd*t+phi)

exact_under_dis = exact_under(times_excat)


C=2
K=1
M=1

ksi = C  / 2 / (K * M) ** .5
w = K/M
wd = w * (1-ksi**2)**.5
print(ksi)
d_c,v,a = advance(t_sim, timestep, n_elem, M, C, K)

a1 = 1
a2 = w
exact_crit = lambda t: a1*np.exp(-w*t) + a2 * t * np.exp(-w*t)

exact_crit_dis = exact_crit(times_excat)

C=3
K=1
M=1

ksi = C  / 2 / (K * M) ** .5
w = K/M
wd = w * (1-ksi**2)**.5
print(ksi)
d_o, v, a = advance(t_sim, timestep, n_elem, M, C, K)


s1 = -(ksi+(ksi**2-1)**.5)*w
s2 = -(ksi-(ksi**2-1)**.5)*w

c1 = s2/(s2-s1)
c2 = -s1/(s2-s1)

exact_over = lambda t: c1 * np.exp(s1*t) + c2*np.exp(s2*t)                                           

#plt.plot(times, d_o.transpose(), '*')
#plt.plot(times_excat, exact_over(times_excat), 'r')

#plt.plot(times, d_c.transpose(), '*')
#plt.plot(times_excat, exact_crit_dis, 'r')
#
#plt.plot(times, d.transpose(), '*')
#plt.plot(times_excat, exact_under_dis,'r')
#


fig, axs = plt.subplots(3)

fig.text(0.5, 0.04, r'Time [s]', ha='center', va='center')
fig.text(0.06, 0.5, r'Displacement: $d$ [mm]', ha='center', va='center', rotation='vertical')


#plt.plot(xnode, np.zeros(n_nodes), '*')
line_exact = axs[2].plot(times_excat, exact_over(times_excat), 'k', label = 'Exact')

line_FE = axs[2].plot(times, d_o.transpose(), '*r', label = r'$\alpha-Method$')

axs[0].plot(times_excat, exact_under_dis, 'k', label='Exact')
axs[0].plot(times, d.transpose(), '*r', label = r'$\alpha$-Method')


axs[1].plot(times_excat, exact_crit_dis, 'k', label='Exact')
axs[1].plot(times, d_c.transpose(), '*r', label = r'$\alpha-Method')

axs[0].text(4.5, 0.8,r'$\xi = 0.5$')
axs[0].legend(loc='upper right')
axs[1].text(4.5, 0.8,r'$\xi = 1$')
axs[2].text(4.5, 0.8,r'$\xi = 1.5$')
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
plt.show()
