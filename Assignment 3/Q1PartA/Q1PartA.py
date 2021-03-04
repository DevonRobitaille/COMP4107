from model_Q1PartA import compileModel, fitModel
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Declare the target function also known as loss
# Input: x-float, y-float
# Output: float between [-1, 1]
loss = lambda x1, x2: np.cos(x1 + 6*0.35*x2) + 2*0.35*x1*x2

interpolate = interp1d([0., 10.], [-1., 1.])

epochs = 10000000

# compileModel(learning_rate, num_hidden_layers, num_neurons)
model2 = compileModel(0.02, 2, 2)
model8 = compileModel(0.02, 2, 8)
model50 = compileModel(0.02, 2, 50)

# fitModel(model, train_size, test_size, batch_size, epochs)
results2, model2, epochs2 = fitModel(model2, 100, 81, 64, epochs)
results8, model8, epochs8 = fitModel(model8, 100, 81, 64, epochs)
results50, model50, epochs50 = fitModel(model50, 100, 81, 64, epochs)

# # x and y are inputs
# # z is the output
print("2 neurons   - test loss: ", results2[0], " epochs: ", epochs2)
print("8 neurons   - test loss: ", results8[0], " epochs: ", epochs8)
print("50 neurons - test loss: ", results50[0], " epochs: ", epochs50)

# Plot the data
X = np.linspace(-1, 1, 10)
Y = np.linspace(-1, 1, 10)

x, y = np.meshgrid(X, Y)

# The loop below is to format the meshgrid to fit the model.predict function
coord = np.zeros((100, 2), dtype=np.float32)
for indexX in range(10):
    for indexY in range(10):
        index = indexX*10 + indexY
        element = [x[indexX, indexY], y[indexX, indexY]]
        coord[index] = element

# Predict values
# target_plot = loss(coord[:, 0], coord[:, 1])
target_plot = loss(x, y)
two_neurons_plot = model2.predict(coord)
eight_neurons_plot  = model8.predict(coord)
fifty_neurons_plot  = model50.predict(coord)

# Correct plot shape
two_neurons_plot = np.reshape(two_neurons_plot, (10, 10))
eight_neurons_plot = np.reshape(eight_neurons_plot, (10, 10))
fifty_neurons_plot = np.reshape(fifty_neurons_plot, (10, 10))

plt.contour(x, y, target_plot, colors='black', label="target")
plt.contour(x, y, two_neurons_plot, colors='red', label="2 neurons")
plt.contour(x, y, eight_neurons_plot, colors='green', label="8 neurons")
plt.contour(x, y, fifty_neurons_plot, colors='blue', label="50 neurons")
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
