import numpy as np
import random

# Separate test and training data set into respective attribute array and class label array
def splitIntoFeaturesAndLabels(test_set, train_set, K, d):

    # Extract all but final column of data set for data attributes
    # Standardize by dividing with element range
    # Extract final column of data set for data class labels
    train_x = np.array(train_set[:, 0:d])
    train_y = np.array(train_set[:, d])
    test_x = np.array(test_set[:, 0:d])
    test_y = np.array(test_set[:, d])

    train_y_binary = np.zeros((len(train_set), K))
    train_y_binary[np.arange(len(train_set)), train_y] = 1

    # Normalize data
    train_x_std = np.std(train_x, axis=0)
    train_x_std[train_x_std == 0] = 0.00000001         # kan jeg tillade mig dette?
    train_x = (train_x - np.mean(train_x, axis=0)) / train_x_std
    test_x = (test_x - np.mean(train_x, axis=0)) / train_x_std     # SKRIV OM DIFF TRAIN and TEST NORMALIZERING

    return train_x, train_y, test_x, test_y, train_y_binary


# Compute Average Accuracy based on set of true and predicted classes
def averageAccuracy(true_classes, pred_classes, K):

    aa = 0
    for i in range(K):
        indices_true = np.where(true_classes == i)
        indices_pred = np.where(pred_classes == i)
        TP = len(np.intersect1d(indices_true, indices_pred))
        FP = np.sum(pred_classes == i) - TP
        FN = np.sum(true_classes == i) - TP
        TN = len(pred_classes) - TP - FP - FN

        aa += ((TP + TN) / float(TP + TN + FP + FN)) / K

    return aa


# Expand input data by including images flipped horizontally
def extendData(data, d):

    # Split data into attributes and labels
    data_x = np.array(data[:, 0:d])
    data_y = np.array(data[:, d])

    image_width = int(np.sqrt(d))
    data_x_flipped = np.zeros(data_x.shape)

    for i in range(data_x.shape[0]):
        image = data_x[i, :].reshape(28, 28)

        # Compute image flipped horizontally (28 x 28)
        flipped = np.zeros((image_width, image_width))
        for j in range(image_width):
            flipped[j, :] = image[j, :][::-1]

        # Convert flipped image into array (1 x 784)
        data_x_flipped[i] = np.array(flipped).flatten()

    # Concatenate original and flipped images
    extended_data_x = np.vstack((data_x, data_x_flipped))
    # Duplicate label array
    extended_data_y = np.hstack((data_y, data_y))
    # Concatenate attributes and labels
    # data_extend = np.hstack((extended_data_x, extended_data_y))
    data_extend = np.zeros((extended_data_x.shape[0], d+1))
    data_extend[:, 0:d] = extended_data_x
    data_extend[:, d] = extended_data_y
    data_extend = data_extend.astype(int)

    return data_extend

data_train = np.load("data/fashion_train.npy")
data_test = np.load("data/fashion_test.npy")

# Shuffle data
random.shuffle(data_train)
random.shuffle(data_test)

# Expand data by its flipped elements
data_train_expand = extendData(data_train, 784)
data_test_expand = extendData(data_test, 784)

train_x, train_y, test_x, test_y, train_y_binary = splitIntoFeaturesAndLabels(data_test_expand, data_train_expand, 5, 784)