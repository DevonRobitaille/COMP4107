from model_Q2PartB import compileModel, fitModel, fitModelNoise

import matplotlib.pyplot as plt

print("Compiling model to be trained on ideal data and noisy data...")
model = compileModel()

print("training model on ideal data...")
modelHist1, model = fitModel(model, 350, 30)
print("training model on noisy data...")
model = fitModelNoise(model, 1)
print("training model on ideal data again...")
modelHist2, model = fitModel(model, 350, 30)

plt.plot(modelHist1.history["loss"])
print(modelHist1.history["accuracy"])
plt.yscale("log")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

plt.plot(modelHist2.history["loss"])
print(modelHist2.history["accuracy"])
plt.yscale("log")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()