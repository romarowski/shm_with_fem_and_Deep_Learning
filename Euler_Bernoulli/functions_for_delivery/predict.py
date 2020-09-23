#Tranining, prediction and performance evaluation of linear percerptron.

import numpy as np
import matplotlib as mpl
from sklearn import linear_model 
from sklearn.metrics import r2_score
from data_gen import simulation
import matplotlib.pyplot as plt

n_elem = 21
node_loc = np.array([1, 3, 5, 7, 10, 12, 14, 16, 18]) #One sensor per node.
E = Iz = L = 1
fixed_dofs = [0, 1, -2, -1] #Fixed-fixed beam.
n_sim = 5000

data  = simulation(n_elem, node_loc, E, Iz, L, fixed_dofs, n_sim) #Train set.
data2 = simulation(n_elem, node_loc, E, Iz, L, fixed_dofs, n_sim) #Test set. 

r2scores = np.zeros(9)

for i in range(9):
    xtrain = data[:, i] #Train on the data of sensor in position i.
    xtrain = xtrain[:, np.newaxis]

    ytrain = data[:, -1] #Label is maximum bending moment.
    ytrain = ytrain[:, np.newaxis]

    reg = linear_model.LinearRegression()

    reg.fit(xtrain, ytrain)


    xpredict = data2[:, i] #Prediction is from sensor i in dataset 2.
    xpredict = xpredict[:, np.newaxis]

    ypredict = reg.predict(xpredict) #Predict moment for input sensors.

    ytest = data2[:, -1]
    ytest = ytest[:, np.newaxis] #Real moment.

    r2scores[i] = r2_score(ytest, ypredict) #Compare prediction with real.


#Plot
plt.stem(np.linspace(0.1, 0.9, 9), r2scores)
plt.yticks([0,1])
plt.xlabel(r'Location of Strain-Gauge')
plt.ylabel(r'$R^2$ score')
mpl.rcParams['axes.grid'] = False
plt.show()


