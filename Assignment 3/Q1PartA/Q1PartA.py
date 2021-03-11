from model_Q1PartA import compileModel, fitModel
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from math import cos
import csv
import random as random

# Declare the target function also known as loss
# Input: x-float, y-float
# Output: float between [-1, 1]
loss = lambda x1, x2: np.cos(x1 + 6*0.35*x2) + 2*0.35*x1*x2

# How many epochs are allowed until the application stops
epochs = 10000000

# Create training data
training_X = np.linspace(-1, 1, 10)
training_Y = np.linspace(-1, 1, 10)
training_x, training_y = np.meshgrid(training_X, training_Y)
inputs_train = np.dstack((training_x, training_y))
inputs_train = inputs_train.reshape(100, 2)
training_z = loss(training_x, training_y)
labels_training = training_z.reshape(100, 1)

# Write the training data and labels to an output file
with open("training_data", 'w') as file:
    for x in range(10):
        for y in range(10):
            index = x*10 +y
            outputStr = str(labels_training[index]) + ":" + str(inputs_train[index, 0]) + "," + str(inputs_train[index, 1]) + "\n"
            file.write(outputStr)

# Create testing data
inputs_testing = np.zeros((81, 2), dtype=np.float32)
labels_testing = np.zeros((81, 1), dtype=np.float32)
for x in range(9):
    for y in range(9):
        index = x*9 +y
        element = [random.uniform(-1, 1), random.uniform(-1, 1)]
        inputs_testing[index] = element
        labels_testing[index] = cos(element[0] + 6*0.35*element[1]) + 2*0.35*element[0]*element[1]
labels_testing = labels_testing.reshape(81,1)

# White the testing data and labels to an output file
with open("testing_data", 'w') as file:
    for x in range(9):
        for y in range(9):
            index = x*9 +y
            outputStr = str(labels_testing[index]) + ":" + str(inputs_testing[index, 0]) + "," + str(inputs_testing[index, 1]) + "\n"
            file.write(outputStr)

# compileModel(learning_rate, num_hidden_layers, num_neurons)
models = []
models.append(compileModel(0.02, 2, 2))
models.append(compileModel(0.02, 2, 8))
models.append(compileModel(0.02, 2, 50))

# fitModel(model, train_size, test_size, batch_size, epochs)
# Write the data to an output file
with open("results_Q1PartA", 'w') as file:
    for index in range(3):
        results, model, total_epochs = fitModel(models[index], inputs_train, labels_training, inputs_testing, labels_testing, 64, epochs)
        file.write("test loss:" + str(results[0]) + ",epochs:" + str(total_epochs) + "\n")

# Create an array to store the z-data for the contour plot diagram
plot_models = []
plot_models.append(loss(training_x, training_y))
for index in range(3):
    plot_models.append(np.reshape(models[index].predict(inputs_train), (10, 10)))

# Add the data to the plot diagram
plt.contour(training_x, training_y, plot_models[0], colors='black', label="target")
plt.contour(training_x, training_y, plot_models[1], colors='red', label="2 neurons")
plt.contour(training_x, training_y, plot_models[2], colors='green', label="8 neurons")
plt.contour(training_x, training_y, plot_models[3], colors='blue', label="50 neurons")

# Display the diagram
proxy = [
    plt.Rectangle((0,0),1,1,fc = "black"),
    plt.Rectangle((0,0),1,1,fc = "red"),
    plt.Rectangle((0,0),1,1,fc = "green"),
    plt.Rectangle((0,0),1,1,fc = "blue")
]
plt.legend(proxy, ["target", "2 neurons", "8 neurons", "50 neurons"], bbox_to_anchor=(1.1, 1.05))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
