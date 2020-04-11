

from functions_for_Keras import *
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import ipdb 
import os


dataset = np.loadtxt('Simulation3.txt')



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

model.add(tf.keras.layers.LSTM(32, 
                               input_shape=x_train_uni.shape[-2:]))

model.add(tf.keras.layers.Dense(1))


model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

EVALUATION_INTERVAL = 500
EPOCHS = 40

model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)
plot_train_history(single_step_history,
                   'Single Step Training and validation loss')
