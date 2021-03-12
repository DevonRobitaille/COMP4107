from model_Q2PartA import compileModel, fitModel, testModelNoise

import matplotlib.pyplot as plt

print("Compiling models with hidden layer size 5-25...")
models = compileModel()

print("Training all models on ideal data...")
modelsHist = []
for model in models:
    modelHist, model = fitModel(model, 350,1)
    modelsHist.append(modelHist)

print("Testing models on both noisy and ideal data...")
recognitionErrors = []
for model in models:
    recognitionErrors.append(testModelNoise(model))

for i in range(len(recognitionErrors)):
    plt.plot(recognitionErrors[i], label="%d nodes" %(i+5))
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.ylabel("recognition error")
plt.xlabel("noise level")
plt.show()
