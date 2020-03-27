
from sklearn import linear_model

from sklearn.metrics import r2_score

import numpy as np

data = np.loadtxt('simulation.txt')

n_sim = 5000
timesteps = 630

#data = data.reshape((timesteps, 2, n_sim))

xtrain = data[0:int(n_sim / 2 * timesteps), 0]
xtrain = xtrain[:, np.newaxis]

ytrain = data[0:int(n_sim / 2 * timesteps), 1]
ytrain = ytrain[:, np.newaxis]

reg = linear_model.LinearRegression()

reg.fit(xtrain, ytrain)

xpredict = data[int(n_sim / 2 + 1):n_sim, 0]
xpredict = xpredict[:, np.newaxis]

ypredict = reg.predict(xpredict)

ytest = data[int(n_sim / 2 + 1):n_sim, 1]
ytest = ytest[:, np.newaxis]

r2 = r2_score(ytest, ypredict)
