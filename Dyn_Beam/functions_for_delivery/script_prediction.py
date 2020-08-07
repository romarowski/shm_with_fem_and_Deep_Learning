from keras import optimizers
from KERAS_utils import *
from keras.models import Sequential
from keras import layers
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import ipdb 
import os
from keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()))


dataset = np.loadtxt('./simulations/simul1.txt')


# 1st normalize the data

nbr_tsteps = np.size(dataset, 0)

TRAIN_SPLIT = int(.8 * nbr_tsteps) #80% for training.
TEST_SPLIT = int(nbr_tsteps - TRAIN_SPLIT)

mean = dataset[:TRAIN_SPLIT].mean(axis=0)
dataset -= mean
std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset /= std

past_history = 50 #.5 seconds into past data displacement data to do 
                          #the prediction.
future_target = 0 #Predict the max moment right after the half a second.

loc_sg = [3, 5] #Set loc of sensor, could be more than one

#Preparate data for tranining and validation.

x_train_, y_train_ = prepare_data(dataset, loc_sg,
                                           0, TRAIN_SPLIT,
                                           past_history,
                                           future_target)
x_val_, y_val_ = prepare_data(dataset, loc_sg,
                                       TRAIN_SPLIT, None,
                                       past_history,
                                       future_target)

tf.random.set_seed(13)

BATCH_SIZE = 256
BUFFER_SIZE = 9600

train_set = tf.data.Dataset.from_tensor_slices((x_train_, y_train_))
train_set = train_set.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_set = tf.data.Dataset.from_tensor_slices((x_val_, y_val_))
val_set = val_set.batch(BATCH_SIZE).repeat()

#Set up topology of Network.
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(16, 
                               input_shape=x_train_.shape[-2:]))

model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mae',
              metrics=[coeff_determination])

EVALUATION_INTERVAL = 200 
EPOCHS = 10 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                  factor=0.2,
                                                  patience=5, 
                                                  min_lr=0.001)

model_history = model.fit(train_set, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_set, 
                          validation_steps = 50,
                          callbacks=[reduce_lr])
