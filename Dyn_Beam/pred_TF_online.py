
from functions_for_Keras import *
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import ipdb 
import os


dataset = np.loadtxt('Simulation4_acel.txt')



#data_dir = '/home/ben/Downloads/jena_climate'
#fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
#f = open(fname)
#data = f.read()
#f.close()
#lines = data.split('\n')
#header = lines[0].split(',')
#lines = lines[1:]
##print(header)
##print(len(lines))
#
#
#dataset = np.zeros((len(lines), len(header) - 1))
#for i, line in enumerate(lines):
#    values = [float(x) for x in line.split(',')[1:]]
#    dataset[i, :] = values
#

#ipdb.set_trace()


#Lets use 90% for training 10% for testing

#plt.plot(data)

# 1st normalize the data

nbr_tsteps = np.size(dataset, 0)

TRAIN_SPLIT = int(.8 * nbr_tsteps)
#nbr_val   = int(.9 * nbr_tsteps)

mean = dataset[:TRAIN_SPLIT].mean(axis=0)
dataset -= mean
std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset /= std

univariate_past_history = 50 
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(dataset, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(dataset, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

tf.random.set_seed(13)

BATCH_SIZE = 256 
BUFFER_SIZE = 9600

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.GRU(32, 
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=x_train_uni.shape[-2:]))
model.add(tf.keras.layers.GRU(64,
                     activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(tf.keras.layers.Dense(1))


model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

EVALUATION_INTERVAL = 500
EPOCHS = 40

model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                     validation_data=val_univariate, validation_steps=50,
                     shuffle=False)

#past_history = 50
#future_target = 0 
#STEP = 1
#
#x_train_single, y_train_single = multivariate_data(dataset[:, 0],
#                                                   dataset[:, 1], 
#                                                   0,
#                                                   TRAIN_SPLIT, 
#                                                   past_history,
#                                                   future_target, 
#                                                   STEP,
#                                                   single_step=True)
#x_val_single, y_val_single = multivariate_data(dataset[:, 0],
#                                               dataset[:, 1],
#                                               TRAIN_SPLIT,
#                                               None, 
#                                               past_history,
#                                               future_target, 
#                                               STEP,
#                                               single_step=True)
#
##ipdb.set_trace()
#
#train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
#
#train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
#
#val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
#
#val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
#
##ipdb.set_trace()
#
#single_step_model = tf.keras.models.Sequential()
#single_step_model.add(tf.keras.layers.LSTM(8,
#                                           input_shape=x_train_single.shape[-2:]))
#single_step_model.add(tf.keras.layers.Dense(1))
#
#single_step_model.compile(optimizer='adam', loss='mae')
##ipdb.set_trace()
#single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
#                                            steps_per_epoch=EVALUATION_INTERVAL,
#                                            validation_data=val_data_single,
#                                            validation_steps=50)
#
#
#single_step_model.save('saved_model/my_model')

#plot_train_history(model,
                   #'Single Step Training and validation loss')
