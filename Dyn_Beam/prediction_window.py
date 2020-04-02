

from sklearn import linear_model

from sklearn.metrics import r2_score

import numpy as np

data = np.loadtxt('Simulation2.txt')

n_sim = 5000
timesteps = 501
window = 501 
nbr_tests = int(np.size(data, 0) / window)

X = np.zeros((nbr_tests, window)) 
j=0
y = np.zeros((nbr_tests , 1))

for i in range(0, np.size(data, 0), window): #check 70 since depends on lenght of simulation
    X[j, :] = data[i:i+window, 0]
    y[j] = data[i+window - 1 , 1] 
    j = j + 1


#data = data.reshape((timesteps, 2, n_sim))

#xtrain = data[0:int(.9 * n_sim * timesteps), 0]
#xtrain = xtrain[:, np.newaxis]

xtrain = X[0:int(.9 * nbr_tests), :]

#ytrain = data[0:int(.9 * n_sim * timesteps), 1]
#ytrain = ytrain[:, np.newaxis]

ytrain = y[0:int(.9 * nbr_tests)]

reg = linear_model.LinearRegression()

reg.fit(xtrain, ytrain)

#xpredict = data[int(.9 * n_sim * timesteps + 1):n_sim * timesteps, 0]
#xpredict = xpredict[:, np.newaxis]

xpredict = X[int(.9 * nbr_tests + 1):nbr_tests, :]

ypredict = reg.predict(xpredict)

#ytest = data[int(.9 * n_sim * timesteps + 1):n_sim * timesteps, 1]
#ytest = ytest[:, np.newaxis]

ytest = y[int(.9 * nbr_tests + 1):nbr_tests]

r2 = r2_score(ytest, ypredict)
