import numpy as np
from random import randint

class HopfieldNetwork():
    def __init__(self, training_set: np.array, bias: int):
        print("CREATING - Hopfield Network")
        self.bias = bias
        self.neurons = training_set.shape[1]
        self.p = training_set.shape[0]
        self.I = np.eye(self.neurons)
        self.W = np.zeros((self.neurons, self.neurons))

        # W = SUM(Xi.(Xi)T)
        for i in range (self.p):
            self.W = np.add(self.W, np.outer(training_set[i], training_set[i]))

        # W = W - P.I
        self.W = np.subtract(self.W, (self.p * self.I))
        print(self.W)
        print("CREATED - Hopfield Network")

    def learn(self, x: np.array):
        index: int = randint(0, self.p)

        #Sum-i = SUM(Uj * Wij)
        mult = np.multiply(self.W[index], x)
        sum = np.sum(mult)
        sign = np.sign(sum)

        # Update the bit towards the energy minima
        x[index] = sign if (sign != 0) else x[index]

        return x, sign