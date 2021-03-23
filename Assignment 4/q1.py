import numpy as np
from hebbian import HopfieldNetwork
from mnist import MNIST

import matplotlib.pyplot as plt

def RunSimulation(mnist, num_of_train: int, num_of_test: int):
    # Build the weighted matrix
    training_set: np.array = mnist.loadTrainingSet(num_of_train)
    hopfield_model = HopfieldNetwork(training_set, 0)

    # start testing
    testing_set, num_of_testing_1 = mnist.loadTestingSet(num_of_test)

    # accuracy
    correct = 0
    total = 0
    
    for i in range (len(testing_set)):
        total = total + 1
        sum = -1
        
        # find energy minima
        prev_loop = np.empty(1)
        curr_loop = testing_set[i]
        while sum != 0 and np.array_equal(prev_loop, curr_loop):     
            testing_set[i], sum = hopfield_model.learn(testing_set[i])
            prev_loop = curr_loop
            curr_loop = testing_set[i]

        # calculate most likely candidate label
        dist_norms = np.empty((2*num_of_train), dtype=np.float)
        for j in range(2*num_of_train):
            # calculate all the distances using np.linalg.norm(target - candidate)
            dist_norms[j] = np.linalg.norm(training_set[j] - testing_set[i])

        # find the mind distance
        min_index = np.argmin(dist_norms)

        # see if the labels match
        if min_index < num_of_train:
            # Min value was a 1
            if i < num_of_testing_1:
                correct = correct + 1
        else:
            # Min value was a 5
            if i >= num_of_testing_1:
                correct = correct + 1

    accuracy: float = (correct/total)
    return accuracy

# Store the mnist data is memory
mnist = MNIST()

# Test accuracy for different amount of training elements
# print(RunSimulation(mnist, 50, 500))
accuracy = []
num_of_test = 1000
for num_of_train in range (1, 50):
    accuracy.append( RunSimulation(mnist, num_of_train, num_of_test) )

# Plot the data (Accuracy vs num of tests)
x = np.arange(1, 50, 1)
y = accuracy

print(y)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(x, y, color='green')
plt.plot(x, y, color='black')
plt.xlabel('Number of Training Cases')
plt.ylabel('Accuracy')
plt.title('Assignment 4 - Question 1')
plt.show()
