from Euler_Bernoulli import *                                           
from scipy import linalg

K = stiffnes(21)                                                        

f = nodal_load(1, 21)                                           

d = linalg.solve(K, f)                                                  

S = stress_recovery(d, 21)     
