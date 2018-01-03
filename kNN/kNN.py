from numpy import array, tile
import operator


def classify(unknown, existing_data, labels, k):
    training_data = array(existing_data)
    size = training_data.shape[0] # Size of training data
    min_vals, max_vals = training_data.min(0), training_data.max(0)  # Column wise minimum and maximum
    ranges = max_vals - min_vals # Feature measurement ranges
    training_data = (known - tile(min_vals, (size, 1))) / tile(ranges, (size, 1))  # Normalize training data
    unknown = (unknown - min_vals) / ranges  # normalize the unknown
    difference = training_data - tile(unknown, (size, 1))  # difference between feature measurements
    squared = difference ** 2  # square the differences
    sq_distances = squared.sum(axis=1)  # sum along the row
    distances = sq_distances ** 0.5  # take the positive square root
    sorted_dist_indexes = distances.argsort()  # Get the indexes of the sorted distances
    votes = {x: 0 for x in set(labels)}  # Label counter
    for i in range(k):
        label = labels[sorted_dist_indexes[i]]  # Get the label based on the index
        votes[label] += 1  # Increment the vote
    return max(votes.items(), key=operator.itemgetter(1))[0]  # Get the label with max votes


data = [x.strip().split() for x in open('datingTestSet.txt').readlines()]
known = [[float(y) for y in x[:-1]] for x in data]
labels = [x[-1] for x in data]
label = classify([40920.0, 8.326976, 0.953952], known, labels, 3)
print(label)

known = [[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]
labels = ['A', 'A', 'B', 'B']
label = classify([0, 0], known, labels, 3)
print(label)
