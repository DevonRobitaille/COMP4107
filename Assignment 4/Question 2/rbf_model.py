import random
import numpy as np

# initializes weights of an rbf network with size hidden_size hidden layer and size output_size output layer
def init_weights(hidden_size, output_size):
    weights = []
    for i in range(output_size):
        outputIWeights = []
        for j in range(hidden_size):
            outputIWeights.append(random.uniform(-1,1))
        weights.append(outputIWeights)

    return weights

# initializes the beta values to be used in the radial basis functions of our hidden layer nodes
def init_betas(clusters, centroids):
    beta_values = []
    for i in range(len(centroids)):
        sum_of_dist = 0
        for j in range(len(clusters[i])):
            sum_of_dist += np.linalg.norm(np.subtract(centroids[i], clusters[i][j][1]))
        sigma = sum_of_dist/len(clusters[i])
        beta = 1/(2 * sigma)
        beta_values.append(beta)

    return beta_values



# performs a set of passed test cases on the network and records accuracy
def test(test_cases, hidden_layer, weights, beta_values):
    correct = 0
    for test_case in test_cases:
        # construct the output of our hidden layer
        hidden_output = []
        for i in range(len(hidden_layer)):
            hidden_output.append(np.exp((-1 * beta_values[i]) * np.linalg.norm(np.subtract(test_case[1], hidden_layer[i]))))

        # construct output of our network
        network_output = []
        for i in range(len(weights)):
            output_i = 0
            for j in range(len(hidden_output)):
                output_i += weights[i][j] * hidden_output[j]
            network_output.append(output_i)

        softmax_output = softmax(network_output)

        # check if the network got the correct classification
        highest = 0
        indexOfHighest = 0
        for i in range(len(softmax_output)):
            if softmax_output[i] > highest:
                highest = softmax_output[i]
                indexOfHighest = i

        if indexOfHighest == test_case[0]:
            correct += 1

    return correct/len(test_cases)



# trains the network on a set of passed training case and performs backpropogation
def train(train_cases, hidden_layer, weights, beta_values):
    for train_case in train_cases:
        # construct the output of our hidden layer
        hidden_output = []
        for i in range(len(hidden_layer)):
            hidden_output.append(np.exp((-1 * beta_values[i]) * np.linalg.norm(np.subtract(train_case[1], hidden_layer[i]))))

        # construct output of our network
        network_output = []
        for i in range(len(weights)):
            output_i = 0
            for j in range(len(hidden_output)):
                output_i += weights[i][j] * hidden_output[j]
            network_output.append(output_i)

        softmax_output = softmax(network_output)

        # create target case
        target = np.zeros(len(weights))
        target[train_case[0]] = 1

        # adjust weights of our network
        for i in range(len(weights)):
            error = target[i] - softmax_output[i]
            for j in range(len(weights[i])):
                weights[i][j] += 0.1 * (error * hidden_output[j])




# performs a set of passed test cases on the network and records accuracy
# implements dropout with probability p
def test_dropout(test_cases, hidden_layer, weights, beta_values, p):
    correct = 0
    for test_case in test_cases:
        # construct the output of our hidden layer
        hidden_output = []
        for i in range(len(hidden_layer)):
            hidden_output.append(np.exp((-1 * beta_values[i]) * np.linalg.norm(np.subtract(test_case[1], hidden_layer[i]))))

        # construct output of our network
        network_output = []
        for i in range(len(weights)):
            output_i = 0
            for j in range(len(hidden_output)):
                output_i += (weights[i][j] * p) * hidden_output[j]
            network_output.append(output_i)

        softmax_output = softmax(network_output)

        # check if the network got the correct classification
        highest = 0
        indexOfHighest = 0
        for i in range(len(softmax_output)):
            if softmax_output[i] > highest:
                highest = softmax_output[i]
                indexOfHighest = i

        if indexOfHighest == test_case[0]:
            correct += 1

    return correct/len(test_cases)



# trains the network on a set of passed training case and performs backpropogation
# adjusts for dropout with probability p
def train_dropout(train_cases, hidden_layer, weights, beta_values, p):
    for train_case in train_cases:
        # create a list of 1s and 0s representing which nodes of our hidden layer we will drop
        dropped = []
        for i in range(len(hidden_layer)):
            if random.random() < p:
                dropped.append(0)
            else:
                dropped.append(1)

        # construct the output of our hidden layer
        hidden_output = []
        for i in range(len(hidden_layer)):
            hidden_output.append(np.exp((-1 * beta_values[i]) * np.linalg.norm(np.subtract(train_case[1], hidden_layer[i]))))

        # apply dropout
        for i in range(len(hidden_output)):
            hidden_output[i] *= dropped[i]

        # construct output of our network
        network_output = []
        for i in range(len(weights)):
            output_i = 0
            for j in range(len(hidden_output)):
                output_i += weights[i][j] * hidden_output[j]
            network_output.append(output_i)

        softmax_output = softmax(network_output)

        # create target case
        target = np.zeros(len(weights))
        target[train_case[0]] = 1

        # adjust weights of our network
        for i in range(len(weights)):
            error = target[i] - softmax_output[i]
            for j in range(len(weights[i])):
                weights[i][j] += 0.1 * (error * hidden_output[j])





# found at https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()