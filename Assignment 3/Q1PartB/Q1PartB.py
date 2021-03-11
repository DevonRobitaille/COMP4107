from model_Q1PartB import compileModel, fitModel
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from math import cos
import csv
import random as random
epochs = 100

def printToFileResults(loss, fileName):
    filename = "loss_per_epoch_" + fileName
    with open(filename, 'w') as file:
        for x in range(epochs):
            outputStr = str(x) + ":" + str(loss[x]) + "\n"
            file.write(outputStr)

# Declare the target function also known as loss
# Input: x-float, y-float
# Output: float between [-1, 1]
loss = lambda x1, x2: np.cos(x1 + 6*0.35*x2) + 2*0.35*x1*x2

# How many epochs are allowed until the application stops


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
model_GDO_8, model_MO_8, model_RMSProp_8 = compileModel(learning_rate, 2, 8)

# fitModel(model, train_size, test_size, batch_size, epochs)
# Fit model for GDO
results_GDO_8, model_GDO_8, data_GDO_8, TimeHistoryResult_GDO_8= fitModel(model_GDO_8, inputs_train, labels_training, inputs_testing, labels_testing, 64, epochs)

# Fit model of MO
results_MO_8, model_MO_8, data_MO_8, TimeHistoryResult_MO_8 = fitModel(model_MO_8, inputs_train, labels_training, inputs_testing, labels_testing, 64, epochs)

# Fit model for RMSProp
results_RMSProp_8, model_RMSProp_8, data_RMSProp_8, TimeHistoryResult_RMSProp_8 = fitModel(model_RMSProp_8, inputs_train, labels_training, inputs_testing, labels_testing, 64, epochs)

# Write the epochs and loss
printToFileResults(data_GDO_8, "GDO_8")

printToFileResults(data_MO_8, "MO_8")

printToFileResults(data_RMSProp_8, "RMSProp_8")

# (2)
x = list(range(0, 100))

fig, ax = plt.subplots()
ax.plot(x, data_GDO_8, label="GDO")
ax.plot(x, data_MO_8, label="MO")
ax.plot(x, data_RMSProp_8, label="RMSProp")
plt.xlabel('epochs')
plt.ylabel('mse')
ax.legend()
plt.show()

# (3)
x = np.arange(99)
fig, ax  = plt.subplots()
ax = fig.add_axes([0,0,1,1])

# Gradient Descent CPU time against epoch
time = [list(i) for i in zip(*TimeHistoryResult_GDO_8)]
y = time[1][1:]
ax.bar(x, y, color = 'b', width = 0.25)

# Momentum CPU time against epoch
time = [list(i) for i in zip(*TimeHistoryResult_MO_8)]
y = time[1][1:]
ax.bar(x + 0.25, y, color = 'g', width = 0.25)

# # RMSProp CPU time against epoch
time = [list(i) for i in zip(*TimeHistoryResult_RMSProp_8)]
y = time[1][1:]
ax.bar(x + 0.5, y, color = 'r', width = 0.25)

ax.set_ylabel('CPU Time (in miliseconds)')
ax.set_xlabel('Epochs')
plt.legend()
plt.show()
GDO_MSE = np.average(tf.keras.losses.MSE( np.reshape(labels_training, (100, 1)), model_GDO_8.predict(inputs_train) ))
MO_MSE = np.average(tf.keras.losses.MSE( np.reshape(labels_training, (100, 1)), model_MO_8.predict(inputs_train) ))
RMSProp_MSE = np.average(tf.keras.losses.MSE( np.reshape(labels_training, (100, 1)), model_RMSProp_8.predict(inputs_train) ))
print("Predict GDO: ", GDO_MSE)
print("Predict MO: ", MO_MSE)
print("Predict RMSProp: ", RMSProp_MSE)

epochs = 500

# Using RMSProp calculate accuracy of different models from 5 to 50 for 500 epochs
models = []
for index in range (5, 25):
    models.append(compileModel(learning_rate, 2, index)[2])

for index in range (len(models)):
    models[index] = fitModel(models[index], inputs_train, labels_training, inputs_testing, labels_testing, 64, epochs)[1]

result = []
for index in range (len(models)):
    result.append(np.average(tf.keras.losses.MSE( np.reshape(labels_training, (100, 1)), models[index].predict(inputs_train) )))

print(result)
labels = np.arange(5, 25, 1)
x = np.arange(len(labels))
fig, ax = plt.subplots()
ax.bar(x, result, 0.25, label='Accuracy')
ax.set_ylabel('Accuracy')
ax.set_xlabel('# of Hidden Neurons')
ax.set_title('Accuracy by hidden neurons')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()
