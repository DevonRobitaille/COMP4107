from k_means import cluster
from rbf_model import init_weights, init_betas, test, train, test_dropout, train_dropout
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

weights = init_weights(k, 10)
beta_values = init_betas(clusters, hidden_layer)

accuracy = []
dropout_chance = 0.5
for i in range(60):
    print("Epoch ", i, "/60")
    train_dropout(digits[(0+(500*i)):(500+(500*i))], hidden_layer, weights, beta_values, dropout_chance)
    accuracy.append(test_dropout(digits[30000:31000], hidden_layer, weights, beta_values, dropout_chance))

plt.plot(accuracy)
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.show()