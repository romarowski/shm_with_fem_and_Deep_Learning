
from keras import optimizers
from functions_for_Keras import *
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


dataset = np.loadtxt('./simulations/Simulation6.txt')


# 1st normalize the data

nbr_tsteps = np.size(dataset, 0)

TRAIN_SPLIT = int(.8 * nbr_tsteps)
TEST_SPLIT = int(nbr_tsteps - TRAIN_SPLIT)
#nbr_val   = int(.9 * nbr_tsteps)
#ipdb.set_trace()
mean = dataset[:TRAIN_SPLIT].mean(axis=0)
dataset -= mean
std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset /= std

univariate_past_history = 50 
univariate_future_target = 0

loc_sg = 4 #[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

x_train_uni, y_train_uni = univariate_data(dataset, loc_sg,
                                           0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)

x_val_uni, y_val_uni = univariate_data(dataset, loc_sg,
                                       TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

tf.random.set_seed(13)

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

#ipdb.set_trace()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(16, return_sequences=True, 
                               input_shape=x_train_uni.shape[-2:]))

model.add(tf.keras.layers.LSTM(8))
model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mae',
              metrics=[coeff_determination])

EVALUATION_INTERVAL = 200
EPOCHS = 100
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
#                                                  factor=0.2,
#                                                  patience=5, 
#                                                  min_lr=0.001)
#
model_history = model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, 
                          validation_steps = 50)#,
                          #callbacks=[reduce_lr])
plot_train_history(model_history,
                   'Single Step Training and validation loss')
