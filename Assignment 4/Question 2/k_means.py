import random
import numpy as np
import matplotlib.pyplot as plt

def display_digit(digit, labeled = True, title = ""):
    """
    graphically displays a 784x1 vector, representing a digit
    """
    if labeled:
        digit = digit[1]
    image = digit
    plt.figure()
    fig = plt.imshow(image.reshape(28,28))
    fig.set_cmap('gray_r')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if title != "":
        plt.title("Inferred label: " + str(title))

def init_centroids(labelled_data,k): ##
    """
    randomly pick some k centers from the data as starting values
    for centroids. Remove labels.
    """
    return list(map(lambda x: x[1], random.sample(labelled_data,k)))

def sum_cluster(labelled_cluster):
    """
    from https://stackoverflow.com/a/20642156
    element-wise sums a list of arrays.
    """
    # assumes len(cluster) > 0
    sum_ = labelled_cluster[0][1].copy()
    for (label,vector) in labelled_cluster[1:]:
        sum_ += vector
    return sum_

def mean_cluster(labelled_cluster): ##
    """
    compute the mean (i.e. centroid at the middle)
    of a list of vectors (a cluster):
    take the sum and then divide by the size of the cluster.
    """
    sum_of_points = sum_cluster(labelled_cluster)
    mean_of_points = sum_of_points * (1.0 / len(labelled_cluster))
    return mean_of_points

def form_clusters(labelled_data, unlabelled_centroids): ##
    """
    given some data and centroids for the data, allocate each
    datapoint to its closest centroid. This forms clusters.
    """
    # enumerate because centroids are arrays which are unhashable
    centroids_indices = range(len(unlabelled_centroids))

    # initialize an empty list for each centroid. The list will
    # contain all the datapoints that are closer to that centroid
    # than to any other. That list is the cluster of that centroid.
    clusters = {c: [] for c in centroids_indices}

    for (label,Xi) in labelled_data:
        # for each datapoint, pick the closest centroid.
        smallest_distance = float("inf")
        for cj_index in centroids_indices:
            cj = unlabelled_centroids[cj_index]
            distance = np.linalg.norm(Xi - cj)
            if distance < smallest_distance:
                closest_centroid_index = cj_index
                smallest_distance = distance
        # allocate that datapoint to the cluster of that centroid.
        clusters[closest_centroid_index].append((label,Xi))
    return list(clusters.values())

def move_centroids(labelled_clusters): ##
    """
    returns list of mean centroids corresponding to clusters.
    """
    new_centroids = []
    for cluster in labelled_clusters:
        new_centroids.append(mean_cluster(cluster))
    return new_centroids

def repeat_until_convergence(labelled_data, labelled_clusters, unlabelled_centroids): ##
    """
    form clusters around centroids, then keep moving the centroids
    until the moves are no longer significant.
    """
    previous_max_difference = 0
    while True:
        unlabelled_old_centroids = unlabelled_centroids
        unlabelled_centroids = move_centroids(labelled_clusters)
        labelled_clusters = form_clusters(labelled_data, unlabelled_centroids)
        # keep old_clusters and clusters so we can get the maximum difference
        # between centroid positions every time.
        differences = map(lambda a, b: np.linalg.norm(a-b),unlabelled_old_centroids,unlabelled_centroids)
        max_difference = max(differences)
        difference_change = abs((max_difference-previous_max_difference)/np.mean([previous_max_difference,max_difference])) * 100
        previous_max_difference = max_difference
        # difference change is nan once the list of differences is all zeroes.
        if np.isnan(difference_change):
            break
    return labelled_clusters, unlabelled_centroids

def cluster(labelled_data, k): ##
    """
    runs k-means clustering on the data.
    """
    centroids = init_centroids(labelled_data, k)
    clusters = form_clusters(labelled_data, centroids)
    final_clusters, final_centroids = repeat_until_convergence(labelled_data, clusters, centroids)
    return final_clusters, final_centroids