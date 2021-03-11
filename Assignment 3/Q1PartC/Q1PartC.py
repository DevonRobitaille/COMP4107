from model_Q1PartC import compileModel, fitModel
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from math import cos
import csv
import random as random

def printToFileResults(loss, fileName):
    filename = "loss_per_epoch_" + fileName
    with open(filename, 'w') as file:
        for x in range(100):
            outputStr = str(x) + ":" + str(loss[x]) + "\n"
            file.write(outputStr)

# Declare the target function also known as loss
# Input: x-float, y-float
# Output: float between [-1, 1]
loss = lambda x1, x2: np.cos(x1 + 6*0.35*x2) + 2*0.35*x1*x2

# How many epochs are allowed until the application stops
epochs = 2500

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

# Write the testing data and labels to an output file
with open("testing_data", 'w') as file:
    for x in range(9):
        for y in range(9):
            index = x*9 +y
            outputStr = str(labels_testing[index]) + ":" + str(inputs_testing[index, 0]) + "," + str(inputs_testing[index, 1]) + "\n"
            file.write(outputStr)

learning_rate = 0.02

# compileModel(learning_rate, num_hidden_layers, num_neurons)
# Compile for 8 hidden neurons
models = []
models.append(compileModel(learning_rate, 2, 8)) # With early stopping
models.append(compileModel(learning_rate, 2, 8)) # Without early stopping

result, model, loss_data = fitModel(models[0], inputs_train, labels_training, inputs_testing, labels_testing, 64, epochs, False)
result_ES, model_ES, loss_data_ES = fitModel(models[1], inputs_train, labels_training, inputs_testing, labels_testing, 64, epochs, True)

# fitModel(model, train_size, test_size, batch_size, epochs)
# Fit model for GDO
plot_models = []
plot_models.append(loss(training_x, training_y))
for index in range(len(models)):
    plot_models.append(np.reshape(models[index].predict(inputs_train), (10, 10)))

# Fig. 7: Function Contours
# Add the data to the plot diagram
plt.contour(training_x, training_y, plot_models[0], colors='black', label="target")
plt.contour(training_x, training_y, plot_models[1], colors='red', label="without early stopping")
plt.contour(training_x, training_y, plot_models[2], colors='green', label="with early stopping")

# Display the diagram
proxy = [
    plt.Rectangle((0,0),1,1,fc = "black"), # target
    plt.Rectangle((0,0),1,1,fc = "red"), # without early stopping
    plt.Rectangle((0,0),1,1,fc = "green"), # with early stopping
]
plt.legend(proxy, ["target", "without early stopping", "with early stopping"], bbox_to_anchor=(1.1, 1.05))
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Fig. 6: Evolution of the MSE
model = compileModel(learning_rate, 2, 8)
validation_mse = []
test_mse = []
training_mse = []
result, model, loss_data = fitModel(model, inputs_train, labels_training, inputs_testing, labels_testing, 64, 1, False)
loss_mse = np.average(tf.keras.losses.MSE( np.reshape(plot_models[0], (100, 1)), model.predict(inputs_train) ))
validation_mse.append(loss_mse)
test_mse.append(result[0])
training_mse.append(loss_data[0])

while loss_mse > 0.02:
    result, model, loss_data = fitModel(model, inputs_train, labels_training, inputs_testing, labels_testing, 64, 1, False)
    loss_mse = np.average(tf.keras.losses.MSE( np.reshape(plot_models[0], (100, 1)), model.predict(inputs_train) ))
    validation_mse.append(loss_mse)
    test_mse.append(result[0])
    training_mse.append(loss_data[0])

result, model, loss_data = fitModel(model, inputs_train, labels_training, inputs_testing, labels_testing, 64, 1, False)
loss_mse = np.average(tf.keras.losses.MSE( np.reshape(plot_models[0], (100, 1)), model.predict(inputs_train) ))
validation_mse.append(loss_mse)
training_mse.append(loss_data[0])
test_mse.append(result[0])

# goal - black
goal_mse = [0.02, 0.02, 0.02]
# test - red
test_mse = test_mse[len(test_mse)-3:]
print(test_mse)
# validation - green
validation_mse = validation_mse[len(validation_mse)-3:]
print(validation_mse)
# training - blue
training_mse = training_mse[len(training_mse)-3:]
print(training_mse)

fig, ax = plt.subplots()
ax.plot([0, 1, 2], goal_mse, label="goal_mse")
ax.plot([0, 1, 2], test_mse, label="test_mse")
ax.plot([0, 1, 2], validation_mse, label="validation_mse")
ax.plot([0, 1, 2], training_mse, label="training_mse")
ax.legend()
plt.show()
