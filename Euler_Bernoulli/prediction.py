import numpy as np

from sklearn import linear_model 

from sklearn.metrics import r2_score

from data_gen import simulation

data, K = simulation(21)

xtrain = data[:, 0]
xtrain = xtrain[:, np.newaxis]

ytrain = data[:, 1]
ytrain = ytrain[:, np.newaxis]

reg = linear_model.LinearRegression()

reg.fit(xtrain, ytrain)

data2 = simulation(21)

xpredict = data2[:, 0]
xpredict = xpredict[:, np.newaxis]

ypredict = reg.predict(xpredict)

ytest = data2[:, 1]
ytest = ytest[:, np.newaxis]

r2 = r2_score(ytest, ypredict)

