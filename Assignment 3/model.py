# References:
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/guide/keras/train_and_evaluate

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from math import floor

# Declare the target function also known as loss
# Input: x-float, y-float
# Output: float between [-1, 1]
loss = lambda x1, x2: np.cos(x1 + 6*0.35*x2) + 2*0.35*x1*x2

#https://stackoverflow.com/questions/51440135/tensorflow-stop-training-when-losses-reach-a-defined-value
class haltCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.02):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")
            self.model.stop_training = True

def compileModel(learning_rate, num_hidden_layers, num_neurons):
    # Build the model
    inputs = keras.Input(shape=(2, ), name="coordinates")
    x = layers.Dense(num_neurons, activation="tanh", name="dense_0")(inputs)
    for index in range(num_hidden_layers-1):
        name = "dense_"+str(index+1)
        x = layers.Dense(num_neurons, activation="tanh", name=name)(x)
    outputs = layers.Dense(1, activation="linear", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Specify training configuration (optimizer, loss, metrics)
    model.compile(
        optimizer = keras.optimizers.SGD( learning_rate ),
        loss = "mse",
        metrics = ["mae", "acc"]
    )

    # return the model we have created
    return model

def fitModel(model, train_size, test_size, batch_size, epochs):
    # Creating training data
    x_train = np.random.uniform(-1, 1, [train_size, 2])
    x_test  = np.random.uniform(-1, 1, [test_size, 2])

    y_train = np.zeros((train_size, 1))
    for index in range(train_size):
        y_train[index] = loss(x_train[index, 0], x_train[index, 1])
    y_test  = np.zeros((test_size, 1))
    for index in range(test_size):
        y_test[index] = loss(x_test[index, 0], x_test[index, 1])

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
        callbacks=[trainingStopCallback]
    )

    # Evaluate the model on the test data
    results = model.evaluate(x_test, y_test, batch_size)

    # return the history of trying to fit the data to the model
    return results, model, len(history.history['loss'])
