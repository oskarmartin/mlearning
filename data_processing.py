import numpy as np

# Separate test and training data set into respective attribute array and class label array
def splitIntoAttAndClass(test_set, train_set, K, d, element_range):

    # Extract all but final column of data set for data attributes
    # Standardize by dividing with element range
    # Extract final column of data set for data class labels
    train_x = np.array(train_set[:, 0:d]) / element_range
    train_y = np.array(train_set[:, d])
    test_x = np.array(test_set[:, 0:d]) / element_range
    test_y = np.array(test_set[:, d])

    train_y_binary = np.zeros((len(train_set), K))
    train_y_binary[np.arange(len(train_set)), train_y] = 1

    # Normalize data
    train_x = (train_x - np.mean(train_x)) / np.std(train_x)
    test_x = (test_x - np.mean(test_x)) / np.std(test_x)

    return train_x, train_y, test_x, test_y, train_y_binary
