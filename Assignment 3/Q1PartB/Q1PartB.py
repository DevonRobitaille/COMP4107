from model_Q1PartB import compileModel, fitModel
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Declare the target function also known as loss
# Input: x-float, y-float
# Output: float between [-1, 1]
# loss = lambda x1, x2: np.cos(x1 + 6*0.35*x2) + 2*0.35*x1*x2

epochs = 100
learning_rate = 0.02

# compileModel(learning_rate, num_hidden_layers, num_neurons)
model_GDO, model_MO, model_RMSProp = compileModel(learning_rate, 2, 8)

# fitModel(model, train_size, test_size, batch_size, epochs)
results_GDO, model_GDO, data_GDO, TimeHistoryResult_GDO = fitModel(model_GDO, 100, 81, 64, epochs)
results_MO, model_MO, data_MO, TimeHistoryResult_MO = fitModel(model_MO, 100, 81, 64, epochs)
results_RMSProp, model_RMSProp, data_RMSProp, TimeHistoryResult_RMSProp = fitModel(model_RMSProp, 100, 81, 64, epochs)

print("GDO - test loss: ", results_GDO[0], " epochs: ", len(data_GDO))
print("MO - test loss: ", results_MO[0], " epochs: ", len(data_MO))
print("RMSProp - test loss: ", results_RMSProp[0], " epochs: ", len(data_RMSProp))


# (2)
x = list(range(0, 100))

# Gradient Descent MSE against epoch
y = data_GDO
plt.plot(x, y)
plt.xlabel('epochs')
plt.ylabel('mse')
plt.show()

# Momentum MSE against epoch
y = data_MO
plt.plot(x, y)
plt.xlabel('epochs')
plt.ylabel('mse')
plt.show()

# RMSProp MSE against epoch
y = data_RMSProp
plt.plot(x, y)
plt.xlabel('epochs')
plt.ylabel('mse')
plt.show()

# (3)
x = np.arange(99)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

# Gradient Descent CPU time against epoch
time = [list(i) for i in zip(*TimeHistoryResult_GDO)]
y = time[1][1:]
y = [int(round(num, 4)*1000) for num in y]
ax.bar(x + 0.0, y, color = 'b', width = 0.25)

# Momentum CPU time against epoch
time = [list(i) for i in zip(*TimeHistoryResult_MO)]
y = time[1][1:]
y = [int(round(num, 4)*1000) for num in y]
ax.bar(x + 0.25, y, color = 'g', width = 0.25)

# RMSProp CPU time against epoch
time = [list(i) for i in zip(*TimeHistoryResult_RMSProp)]
y = time[1][1:]
y = [int(round(num, 4)*1000) for num in y]
ax.bar(x + 0.5, y, color = 'r', width = 0.25)

ax.set_ylabel('CPU Time (in miliseconds)')
ax.set_xlabel('Epochs')
ax.set_yticks(np.arange(15, 25, 10))
ax.legend(labels=['GDO', 'MO', 'RMSProp'])
plt.show()
