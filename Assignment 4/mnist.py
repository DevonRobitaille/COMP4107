import numpy as np


class MNIST:
    def __init__(self):
        print("LOADING - MNIST dataset")

        self.data_path = "mnist/" 
        self.dataForTraining = np.loadtxt(self.data_path + "mnist_train.csv", delimiter=",", dtype=int)
        self.dataForTesting = dataForTesting = np.loadtxt(self.data_path + "mnist_test.csv",  delimiter=",", dtype=int)

        # Prepare training data
        self.training_1 = self.dataForTraining[ (self.dataForTraining[:,0] == 1) ]
        self.training_1 = self.training_1[:, 1:]

        self.training_5 = self.dataForTraining[ (self.dataForTraining[:,0] == 5) ]
        self.training_5 = self.training_5[:, 1:]

        # Prepare testing data
        self.testing_1 = self.dataForTesting[ (self.dataForTesting[:,0] == 1) ]
        self.testing_1 = self.testing_1[:, 1:]

        self.testing_5 = self.dataForTesting[ (self.dataForTesting[:,0] == 5) ]
        self.testing_5 = self.testing_5[:, 1:]

        print("LOADED  - MNIST dataset")

    def loadTrainingSet(self, num_of_train: int):
        t_1 = np.copy(self.training_1)
        t_1[t_1 <= 127] = -1
        t_1[t_1 > 127] = 1

        t_5 = np.copy(self.training_5)
        t_5[t_5 <= 127] = -1
        t_5[t_5 > 127] = 1

        return np.concatenate((t_1[:num_of_train], t_5[:num_of_train]), axis=0)

    def loadTestingSet(self, num_of_test: int):
        t_1 = np.copy(self.testing_1)
        t_1[t_1 <= 127] = -1
        t_1[t_1 > 127] = 1

        t_5 = np.copy(self.testing_5)
        t_5[t_5 <= 127] = -1
        t_5[t_5 > 127] = 1

        return np.concatenate((t_1[:num_of_test], t_5[:num_of_test]), axis=0), len(t_1[:num_of_test])
