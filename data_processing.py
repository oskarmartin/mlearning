import numpy as np

# Separate test and training data set into respective attribute array and class label array
def splitIntoFeaturesAndLabels(test_set, train_set, K, d, element_range):

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
    print(train_x_std)
    train_x_std[train_x_std == 0] = 0.00000001         # kan jeg tillade mig dette?
    train_x = (train_x - np.mean(train_x, axis=0)) / train_x_std
    test_x = (test_x - np.mean(train_x, axis=0)) / train_x_std     # SKRIV OM DIFF TRAIN and TEST NORMALIZERING

    return train_x, train_y, test_x, test_y, train_y_binary


# Average accuracy:
# Compute average accuracy based on set of true and predicted classes
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
