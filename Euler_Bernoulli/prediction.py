import numpy as np

import matplotlib as mpl

from sklearn import linear_model 

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_gen import simulation

import matplotlib.pyplot as plt

n_elem = 21


node_loc = np.array([1, 3, 5, 7, 10, 12, 14, 16, 18])

data, K = simulation(n_elem, node_loc)

data2, K = simulation(n_elem, node_loc)

r2scores = np.zeros(9)
mses = np.zeros(9)
maes = np.zeros(9)
rmses = np.zeros(9)

for i in range(9):
    xtrain = data[:, i]
    xtrain = xtrain[:, np.newaxis]

    ytrain = data[:, -1]
    ytrain = ytrain[:, np.newaxis]

    reg = linear_model.LinearRegression()

    reg.fit(xtrain, ytrain)


    xpredict = data2[:, i]
    xpredict = xpredict[:, np.newaxis]

    ypredict = reg.predict(xpredict)

    ytest = data2[:, -1]
    ytest = ytest[:, np.newaxis]

    r2scores[i] = r2_score(ytest, ypredict)
    mses[i] = mean_squared_error(ytest, ypredict)
    maes[i] = mean_absolute_error(ytest, ypredict)
    rmses[i] = mean_squared_error(ytest, ypredict, squared=False)
    

#plt.stem(np.linspace(0.1, 0.9, 9), r2scores)
#plt.yticks([0,1])
#plt.xlabel(r'Location of Strain-Gauge')
#plt.ylabel(r'$R^2$ score')
#mpl.rcParams['axes.grid'] = False
#plt.show()


################
#Plotting for sensor at x=.5L

i=4
xtrain = data[:, i]
xtrain = xtrain[:, np.newaxis]

ytrain = data[:, -1]
ytrain = ytrain[:, np.newaxis]

reg = linear_model.LinearRegression()

reg.fit(xtrain, ytrain)


xpredict = data2[:, i]
xpredict = xpredict[:, np.newaxis]

ypredict = reg.predict(xpredict)

ytest = data2[:, -1]
ytest = ytest[:, np.newaxis]



print(r2_score(ytest, ypredict))
#f = plt.figure()
##plt.plot(np.arange(10), ytest[100:110], 'bx', label='True')
#plt.plot(np.arange(10), ypredict[100:110], 'ro', fillstyle='none', label='Predicted')
#
#plt.grid()
#plt.xlabel('Simulation')
#plt.ylabel('Bending moment [Nm]')
#plt.legend()
#plt.show()
#
#f.savefig("foo.png", bbox_inches='tight', dpi=600)
f1 = plt.figure()
plt.plot(xpredict[100:110]*1000, ytest[100:110], 'bx', label='True')
plt.plot(xpredict[100:110]*1000, ypredict[100:110], 'ro', fillstyle='none', label='Predicted')
plt.legend()
plt.xlabel('Displacement [mm]')
plt.ylabel('Bending moment [Nm]')

f1.savefig("linear2.png", bbox_inches='tight', dpi=600)
#generate multiple r2 scores for multiple positions of the sg 

#
#
#r2scores = np.zeros((9, 200))
#
#for i, node in enumerate(node_loc):
    #    for sim in np.arange(r2scores.shape[1]): 
    #        data3, K = simulation(n_elem, loc_sg)
#                
#        xpredict = data3[:, 0]
#        xpredict = xpredict[:, np.newaxis]
#
#        ypredict = reg.predict(xpredict)
#
#        ytest = data3[:, 1]
#        ytest = ytest[:, np.newaxis]
#        
#        r2scores[i, sim] = r2_score(ytest, ypredict) 
#    
