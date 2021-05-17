from keras import optimizers
from KERAS_utils import *
from keras.models import Sequential
from keras import layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop, Adam
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import ipdb 
import os
from keras import backend as K


import datetime 

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()))


dataset = np.loadtxt('./simulations/random_simul2.txt')
dataset2 = np.loadtxt('./simulations/simul1.txt')


datase = np.append(dataset, dataset2, axis=0)

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

loc_sg = [1, 7] #[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

x_train_uni, y_train_uni = prepare_data(dataset, loc_sg,
                                           0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)

x_val_uni, y_val_uni = prepare_data(dataset, loc_sg,
                                       TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

tf.random.set_seed(13)

BATCH_SIZE = 256
BUFFER_SIZE = 400000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()


val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

#ipdb.set_trace()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(2,  
                               input_shape=x_train_uni.shape[-2:]))

#model.add(tf.keras.layers.LSTM(32, return_sequences = True))
#model.add(tf.keras.layers.LSTM(16, return_sequences = True))
#model.add(tf.keras.layers.LSTM(8))
model.add(tf.keras.layers.Dense(1))

model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mse',
              metrics=[coeff_determination,
                      tf.keras.metrics.MeanSquaredError(),
                      tf.keras.metrics.RootMeanSquaredError(),
                      tf.keras.metrics.MeanAbsoluteError()])

EVALUATION_INTERVAL = 300 
EPOCHS = 30

#-----------CALLBACKS-----------------------------------------
path_checkpoint = 'checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.1,
                                                 patience=0, 
                                                 min_lr=0.001,
                                                 verbose=1)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5,verbose=1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [callback_checkpoint,
             callback_early_stopping,
             callback_tensorboard,
             callback_reduce_lr]
#-------------------------------------------------------------

model_history = model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, 
                          validation_steps = 50,
                          callbacks=callbacks)

#plot_train_history(model_history,
 #                  'Single Step Training and validation loss')