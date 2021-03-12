from model_Q2PartC import compileModel, fitModel, testModelNoise, fitModelNoise

import matplotlib.pyplot as plt

print("Compiling model to be trained on ideal data only...")
model = compileModel()
print("Compiling model to be trained on ideal data and noisy data...")
modelNoise = compileModel()

print("training model 1 on ideal data...")
model, modelHist = fitModel(model, 350, 30)
print("training model 2 on ideal data...")
modelNoise, modelNoiseHist1 = fitModel(modelNoise, 350, 30)
print("training model 2 on noisy data...")
modelNoise = fitModelNoise(modelNoise, 1)
print("training model 2 on ideal data again...")
modelNoise, modelNoiseHist2 = fitModel(modelNoise, 350, 30)

print("Testing both models ability to recognize noisy and non-noisy data...")
recognitionErrors = []
recognitionErrors.append(testModelNoise(model))
recognitionErrors.append(testModelNoise(modelNoise))

plt.plot(recognitionErrors[0], label="Trained without noise")
plt.plot(recognitionErrors[1], label="Trained with noise")
plt.legend(loc='upper left')
plt.ylabel("recognition error")
plt.xlabel("noise level")
plt.show()