from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

data = np.loadtxt('Simulation2.txt')

#Lets use 90% for training 10% for testing

# 1st normalize the data

nbr_tsteps = np.size(data, 0)

nbr_train = .9 * nbr_tsteps

mean = data[:nbr_train].mean(axis=0)
data -= mean
std = data[:nbr_train].std(axis=0)
data /= std

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop, loss='mae')
history = model



def generator(data, lookback, delay):

