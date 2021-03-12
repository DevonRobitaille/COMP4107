import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import copy

trainSetX = []
with open('Q2_data.txt') as f:
    for line in f:
        trainSetX.append([int(x) for x in line.split()])

trainSetY = np.zeros((31, 31), dtype="int")
for i in range(31):
    trainSetY[i][i] = 1



def compileModel():
    # Build an array of models of varying hidden layer size
    inputs = keras.Input(35, name="image")
    x = layers.Dense(23,
                     activation="sigmoid",
                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                     name="dense_0")(inputs)
    outputs = layers.Dense(31,
                           activation="softmax",
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                           name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Specify training configuration (optimizer, loss)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics="accuracy"
    )

    return model


def fitModel(model, epochs, train_size):
    # Construct the training set
    x_train = np.array(trainSetX)
    y_train = trainSetY
    for i in range(train_size):
        x_train = np.vstack((x_train, np.array(trainSetX)))
        y_train = np.vstack((y_train, trainSetY))

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs
    )

    return model, history

def fitModelNoise(model, epochs):
    # Construct the training set
    for trainingCount in range(10): # number of times to train the network on noisy data

        for noiseLevel in range(3):
            x_train_set = []
            y_train_set = []

            # generate a training set with flipped bits equal to noise_level
            for i in range(1): # size of each training set

                # generate one "block" of our training set
                x_train = copy.deepcopy(trainSetX)
                y_train = np.ndarray.tolist(trainSetY)
                for row in x_train:
                    noise = random.sample(np.arange(len(row)).tolist(), noiseLevel + 1)
                    for x in noise:
                        if row[x] == 0:
                            row[x] = 1
                        else:
                            row[x] = 0

                # append our "block" to the final training set
                x_train_set += x_train
                y_train_set += y_train

            model.fit(
                x_train_set,
                y_train_set,
                epochs=epochs
            )
    return model


def testModelNoise(model):
    recognitionError = []
    for noiseLevel in range(4):
        x_test_set = []
        y_test_set = []

        # generate a test set with flipped bits equal to noise_level
        for i in range(1000): # size of each test set. value is high for accurate measurment

            # generate one "block" of our test set
            x_test = copy.deepcopy(trainSetX)
            y_test = np.ndarray.tolist(trainSetY)
            for row in x_test:
                noise = random.sample(np.arange(len(row)).tolist(), noiseLevel)
                for x in noise:
                    if row[x] == 0:
                        row[x] = 1
                    else:
                        row[x] = 0

            # append our "block" to the final test set
            x_test_set += x_test
            y_test_set += y_test

        history = model.evaluate(x_test_set, y_test_set)
        recognitionError.append(1 - history[1])

    return recognitionError

