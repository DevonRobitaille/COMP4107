# References:
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/guide/keras/train_and_evaluate

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from math import floor

#https://stackoverflow.com/questions/51440135/tensorflow-stop-training-when-losses-reach-a-defined-value
class haltCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.02):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")
            self.model.stop_training = True

# Description: Creates a model (the shape of the neural network)
# Inputs:
#   learning_rate = float
#   num_hidden_layers = int
#   num_neurons = int
# Outputs:
#   model = model
def compileModel(learning_rate, num_hidden_layers, num_neurons):
    # Build the model
    inputs = keras.Input(shape=(2, ), name="coordinates")
    x = layers.Dense(num_neurons, activation="tanh", name="dense_0")(inputs)
    for index in range(num_hidden_layers-1):
        name = "dense_"+str(index+1)
        x = layers.Dense(num_neurons, activation="tanh", name=name)(x)
    outputs = layers.Dense(1, activation="linear", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    RMSProp_Opt = tf.compat.v1.train.RMSPropOptimizer( learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, name='RMSProp' )

    # Specify training configuration (optimizer, loss, metrics)
    model.compile(
        optimizer = RMSProp_Opt,
        loss = "mse",
        metrics = ["mae", "acc"]
    )

    # return the model we have created
    return model

# Description: Fit the model against a selection of data, and then evaluate/verify the model
# Inputs:
#   model = model
#   x_train = array containing inputs values for the model
#   y_train = array containing labels for input data
#   x_test = array containing inputs values for the model
#   y_test = array containing labels for input data
#   batch_size = int
#   epochs = int
# Outputs:
#   model = model
def fitModel(model, x_train, y_train, x_test, y_test, batch_size, epochs, set_Early_Stopping):
    trainingStopCallback = haltCallback()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_split = 0.2,
        callbacks= [trainingStopCallback] if (set_Early_Stopping) else []
    )

    # Evaluate the model on the test data
    results = model.evaluate(x_test, y_test, batch_size)

    # return the history of trying to fit the data to the model
    return results, model, history.history['loss']
