import numpy as np

def loadTrainingSet(num_of_train: int):
    # Step 0 - Load MNIST data

    print("LOADING - MNIST dataset")

    data_path = "mnist/"
    dataForTraining = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")

    print("LOADED  - MNIST dataset")

    label_1 = dataForTraining[ (dataForTraining[:,0] == 1) ]
    label_1 = label_1[:, 1:]
    label_1[label_1 > 127] = 1
    label_1[label_1 <= 127] = -1
    print(label_1)

    label_5 = dataForTraining[ (dataForTraining[:,0] == 1) ]
    label_5 = label_5[:, 1:]
    label_5[label_5 > 127] = 1
    label_5[label_5 <= 127] = -1

    return np.concatenate((label_1[:num_of_train], label_5[:num_of_train]), axis=0)

def loadTestingSet(num_of_test: int):
    # Step 0 - Load MNIST data

    print("LOADING - MNIST dataset")

    data_path = "mnist/"
    dataForTesting = np.loadtxt(data_path + "mnist_test.csv",  delimiter=",")

    print("LOADED  - MNIST dataset")

    label_1 = dataForTesting[ (dataForTesting[:,0] == 1) ]
    label_1 = label_1[:, 1:]
    label_1[label_1 > 127] = 1
    label_1[label_1 <= 127] = -1

    label_5 = dataForTesting[ (dataForTesting[:,0] == 1) ]
    label_5 = label_5[:, 1:]
    label_5[label_5 > 127] = 1
    label_5[label_5 <= 127] = -1

    return np.concatenate((label_1[:num_of_test], label_5[:num_of_test]), axis=0)
