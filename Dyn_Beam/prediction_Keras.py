#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


from functions_for_Keras import generator
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
import numpy as np
import ipdb 
import os


#data = np.loadtxt('Simulation4_acel.txt')



data_dir = '/home/ben/Downloads/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
#print(header)
#print(len(lines))


float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values





#Lets use 90% for training 10% for testing

#plt.plot(data)

# 1st normalize the data

nbr_tsteps = np.size(float_data, 0)

nbr_train = 200000
#nbr_val   = int(.9 * nbr_tsteps)

mean = float_data[:nbr_train].mean(axis=0)
float_data -= mean
std = float_data[:nbr_train].std(axis=0)
float_data /= std


lookback = 1440
step = 6 
delay = 144 
batch_size = 128

#ipdb.set_trace()

train_gen = generator(float_data,
                      lookback=lookback, 
                      delay=delay,
                      min_index=0,
                      max_index=200e3, 
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)
#next(train_gen)
#next(train_gen)
ipdb.set_trace()
val_gen = generator(float_data,
                    lookback=lookback, 
                    delay=delay,
                    min_index=200001, 
                    max_index=300e3,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(float_data, 
                    lookback=lookback,
                    delay=delay,
                    min_index=300001,
                    max_index=None,
                    step=step,
                    batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)

test_steps = (len(float_data) - 300001 - lookback)





model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, 
                              epochs=2, 
                              validation_data=val_gen,
                              validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend

plt.show()




