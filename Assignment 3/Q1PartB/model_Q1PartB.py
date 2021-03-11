import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import time

TimeHistoryResult = []

class TimeHistory(keras.callbacks.Callback):
        def __init__(self):
            self.times = []
        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()
        def on_epoch_end(self, epoch, logs={}):
            self.times.append((epoch, time.time() -  self.epoch_time_start))
        def on_train_end(self, logs = {}):
            TimeHistoryResult = self.times

# Declare the target function also known as loss
# Input: x-float, y-float
# Output: float between [-1, 1]
loss = lambda x1, x2: np.cos(x1 + 6*0.35*x2) + 2*0.35*x1*x2

def compileModel(learning_rate, num_hidden_layers, num_neurons):
    # Build the models
    inputs = keras.Input(shape=(2, ), name="coordinates")
    x = layers.Dense(num_neurons, activation="tanh", name="dense_0")(inputs)
    for index in range(num_hidden_layers-1):
        name = "dense_"+str(index+1)
        x = layers.Dense(num_neurons, activation="tanh", name=name)(x)
    outputs = layers.Dense(1, activation="linear", name="predictions")(x)

    model_GDO = keras.Model(inputs=inputs, outputs=outputs)
    model_MO = keras.Model(inputs=inputs, outputs=outputs)
    model_RMSProp = keras.Model(inputs=inputs, outputs=outputs)

    # traingd w/ GradientDescentOptimizer
    GDO_Opt = tf.compat.v1.train.GradientDescentOptimizer( learning_rate, use_locking=False, name='GradientDescent' )

    model_GDO.compile(
        optimizer = GDO_Opt,
        loss = "mse",
        metrics = ["mae", "acc"]
    )


    # traingdm w/ MomentumOptimizer
    MO_Opt = tf.compat.v1.train.MomentumOptimizer( learning_rate, 0, use_locking=False, name='Momentum', use_nesterov=False )

    model_MO.compile(
        optimizer = MO_Opt,
        loss = "mse",
        metrics = ["mae", "acc"]
    )


    # traingrms w/ RMSPropOptimizer
    RMSProp_Opt = tf.compat.v1.train.RMSPropOptimizer( learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, name='RMSProp' )

    model_RMSProp.compile(
        optimizer = RMSProp_Opt,
        loss = "mse",
        metrics = ["mae", "acc"]
    )

    return model_GDO, model_MO, model_RMSProp

def fitModel(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    timetaken = TimeHistory()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_split = 0.2,
        callbacks = [timetaken]
    )

    # Evaluate the model on the test data
    results = model.evaluate(x_test, y_test, batch_size)

    # return the history of trying to fit the data to the model
    return results, model, history.history['loss'], timetaken.times
