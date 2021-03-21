import numpy as np
from hebbian import HopfieldNetwork
from mnist import loadTrainingSet, loadTestingSet

def main(num_of_train: int, num_of_test: int):
    # Build the weighted matrix
    training_set: np.array = loadTrainingSet(num_of_train)
    hopfield_model = HopfieldNetwork(training_set, 0)

    # start testing
    testing_set: np.array = loadTestingSet(num_of_test)

    print("Checking 1's")
    # start with 1's
    for i in range(num_of_test):
        sum = -1
        
        # find energy minima
        prev_loop = np.empty(1)
        curr_loop = testing_set[i]
        while sum != 0 and np.array_equal(prev_loop, curr_loop):     
            testing_set[i], sum = hopfield_model.learn(testing_set[i])
            prev_loop = curr_loop
            curr_loop = testing_set[i]

        # find which label is matches
        for j in range(2*num_of_train):
            if np.array_equal(testing_set[i], training_set[j]):
                print("Guess: 1 - Match: ", ("1" if j < num_of_train else "5"))
                break

    print("Checking 5's")
    # start with 1's
    for i in range(num_of_test, 2*num_of_test):
        sum = -1
        
        # find energy minima
        prev_loop = np.empty(1)
        curr_loop = testing_set[i]
        while sum != 0 and np.array_equal(prev_loop, curr_loop):     
            testing_set[i], sum = hopfield_model.learn(testing_set[i])
            prev_loop = curr_loop
            curr_loop = testing_set[i]

        # find which label is matches
        for j in range(2*num_of_train):
            if np.array_equal(testing_set[i], training_set[j]):
                print("Guess: 5 - Match: ", ("1" if j < num_of_train else "5"))
                break

num_of_train = 1
num_of_test = 20
main(num_of_train, num_of_test)
