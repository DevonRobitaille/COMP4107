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
    correct_label_1 = 0
    correct_label_5 = 0
    total_label_1 = 0
    total_label_5 = 0
    
    for i in range (len(testing_set)):
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

        # find the min distance
        min_index = np.argmin(dist_norms)

        # see if the labels match
        if i < num_of_testing_1:
            total_label_1 = total_label_1 + 1
            if (min_index < num_of_train):
                correct_label_1 = correct_label_1 + 1
        else:
            # Min value was a 5
            total_label_5 = total_label_5 + 1
            if (min_index >= num_of_train):
                correct_label_5 = correct_label_5 + 1

    # print("   1 - Correct: %d, Total: %d" % (correct_label_1, total_label_1))
    # print("   5 - Correct: %d, Total: %d" % (correct_label_5, total_label_5))
    accuracy: float = ((correct_label_1 + correct_label_5)/(total_label_1 + total_label_5))
    return accuracy

# Store the mnist data is memory
mnist = MNIST()

# Test accuracy for different amount of training elements
accuracy = []
num_of_test = 500
for num_of_train in range (1, 100):
    result = RunSimulation(mnist, num_of_train, num_of_test)
    accuracy.append( result )

# Plot the data (Accuracy vs num of tests)
x = np.arange(1, 100, 1)
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
