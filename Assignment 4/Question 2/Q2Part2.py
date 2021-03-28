from k_means import cluster
from rbf_model import init_weights, init_betas, test, train
from base64 import b64decode
from json import loads
import numpy as np
import matplotlib.pyplot as plt

def read_in_data(json_line):
    """
    to parse the a line of the digits file into tuples of
    (labelled digit, numpy array of vector representation of digit)
    """
    json_object = loads(json_line)
    json_data = b64decode(json_object["data"])
    digit_vector = np.fromstring(json_data, dtype=np.ubyte)
    digit_vector = digit_vector.astype(np.float64)
    return (json_object["label"], digit_vector)


with open("digits.base64.json","r") as f:
    data = []
    for i in range(60000): # This is the size of the dataset we are loading in. Value can be as high as 60000 to load all data.
        data.append(f.readline())
    digits = list(map(read_in_data, data))

k_means_training = digits[59000:] # the set of data used to perform k-means clustering

k = 20
print("starting clustering for k =", k)
clusters, hidden_layer = cluster(k_means_training, k)
print("ending clustering")

k_fold = 5
weights = []
beta_values = []
datasets = []
for i in range(k_fold):
    weights.append(init_weights(k, 10))
    beta_values.append(init_betas(clusters, hidden_layer))
    datasets.append(digits[i * 5000:(i + 1) * 5000])


performance = []
for i in range(len(datasets)):
    print("Performing cross validation ", i, " of ", len(datasets))

    # construct the current training set
    train_set = []
    for j in range(len(datasets)):
        if i != j:
            train_set += datasets[j]

    # train and test the network
    accuracy = []
    for j in range(40):
        print("Epoch ", j, "/40")
        train(train_set[(0+(500*j)):(500+(500*j))], hidden_layer, weights[i], beta_values[i])
        accuracy.append(test(datasets[i], hidden_layer, weights[i], beta_values[i]))
    performance.append(accuracy)


print("Calculating the mean accuracy of each network after 40 epochs...")
final_acc = []
for acc in performance:
    final_acc.append(acc[len(acc) - 1])
print("The mean accuracy across the 5 networks is: ", np.mean(final_acc))
print("Calculating the standard deviation of accuracy across the 5 networks...")
print("The standard deviation of accuracy across the 5 networks is: ", np.std(final_acc))


# display each network accuracy
for accuracy in performance:
    plt.plot(accuracy, label="Cross validation set")
plt.ylabel("accuracy")
plt.xlabel("Epochs")
plt.show()


average_accuracy = []
for i in range(len(performance[i])):
    sum = 0
    for j in range(len(performance)):
        sum += performance[j][i]
    average_accuracy.append(sum/len(performance))

# display the average network accuracy
plt.plot(average_accuracy)
plt.ylabel("accuracy")
plt.xlabel("Epochs")
plt.show()