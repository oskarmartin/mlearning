import numpy as np
import random
import matplotlib.pyplot as plt
import ml_tools as ml
import lda_plots as lplots


# Reduce dimensionality of input data using Linear Discriminant Analysis
def reduceDimensionsUsingLDA(X, y, K, d):

    # Overall data mean by dimension
    data_dim_mean = np.mean(X, axis=0).reshape((d, 1))

    # Compute class mean by dimension m_i
    class_size = np.zeros(K).astype(int)
    class_dim_mean = []
    S_W = np.zeros((d, d))
    for i in range(K):
        class_i = X[y == i]
        class_size[i] = class_i.shape[0]
        class_dim_mean.append(np.mean(class_i, axis=0))
        m_i = class_dim_mean[i].reshape(d, 1)

        # Within-class scatter matrix S_i
        S_i = np.zeros((d, d))
        for n in range(class_size[i]):
            x_t = class_i[n].reshape((d, 1))
            S_i += (x_t - m_i).dot((x_t - m_i).transpose())

        # Total within-class scatter matrix S_W
        S_W += S_i

    # Between class scatter matrix S_B
    S_B = np.zeros((d, d))
    for i in range(K):
        m_i = class_dim_mean[i].reshape(d, 1)       # Class mean by dimension
        S_B += class_size[i] * (m_i - data_dim_mean).dot((m_i - data_dim_mean).transpose())

    # J for maximization
    J = np.linalg.inv(S_W).dot(S_B)

    # Sort eigenvectors of J according to non-increasing eigenvalues
    eigValues, eigVectors = np.linalg.eig(J)
    eigValues = np.real(eigValues)
    eigVectors = np.real(eigVectors)
    eigVectors_sorted = eigVectors.transpose()[np.argsort(eigValues)[::-1]]

    # Compute for each eigenvector the variance explained
    eigValues_sorted = np.sort(eigValues)[::-1]
    var_explained = (eigValues_sorted / np.sum(eigValues_sorted)) * 100

    # Project data onto the eigenvectors of J and extract the K-1 relevant dimensions
    Z = eigVectors_sorted.dot(X.transpose())

    return Z, var_explained, eigVectors_sorted


# Compute decision boundary weights of input data using Least Square Classifier
def performLeastSquareClassifier(X, T):

    # Add 1-column to training data for decision boundary intercept
    d = X.shape[1]
    X_extended = np.ones((X.shape[0], d + 1))
    X_extended[:, 0:d] = X

    # Estimate decision boundary weights
    W = np.linalg.inv((X_extended.transpose()).dot(X_extended)).dot((X_extended.transpose()).dot(T))

    return W


# Compare predicted labels to true labels and output Average Accuracy
def predictLabels(test_data_x, true_labels, trained_weights, K):

    # Add 1-column to testing data to accommodate the added decision boundary intercept
    d = test_data_x.shape[1]
    test_data_extended = np.ones((test_data_x.shape[0], d + 1))
    test_data_extended[:, 0:d] = test_data_x

    # Estimate data labels and compute performance
    probability_y = test_data_extended.dot(trained_weights)
    y = np.argmax(probability_y, axis=1)
    ave_acc = ml.averageAccuracy(true_labels, y, K)

    return ave_acc


# Plot least squares Average Accuracy as a function of LDA dimensions included in training
def plotPerformanceOverDimensions(lda_ave_acc, x_ticks, title, D):

    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(D-1), lda_ave_acc, color='tab:blue')
    plt.title(title, size=13)
    plt.xticks(np.arange(D-1), x_ticks, size=9)
    plt.xlabel("\nLDA dimensions included in training the model\n", size=11)
    plt.ylabel("Average testing accuracy\n", size=11)
    fig.savefig('projectionPlots/least_squares_performance.png')

    plt.show()


# Use 2 and 3 LDA dimensions to call plotting of data with and without decision boundaries
def initiateCreationOfLDAPlots(lda_train, lda_test, train_y_binary, train_y, test_y, var_explained, K):

    lda_train_2D = lda_train[0:2, :]
    lda_train_3D = lda_train[0:3, :]

    lda_test_2D = lda_test[0:2, :]
    lda_test_3D = lda_test[0:3, :]

    W_2D = performLeastSquareClassifier(lda_train_2D.transpose(), train_y_binary)
    W_3D = performLeastSquareClassifier(lda_train_3D.transpose(), train_y_binary)

    x_label = 've = {:0.2f}%'.format(var_explained[0])
    y_label = 've = {:0.2f}%'.format(var_explained[1])
    z_label = 've = {:0.2f}%'.format(var_explained[2])

    ml.plotDataIn2D(lda_train_2D, train_y, K, x_label, y_label, 'LDA', 'training')
    ml.plotDataIn3D(lda_train_3D, train_y, K, x_label, y_label, z_label, 'LDA', 'training', 30, 45)

    lplots.plot2DwithDecisionsLDA(W_2D, lda_train_2D, train_y, K, var_explained, 'training')
    lplots.plot3DwithDecisionsLDA(W_3D, lda_train_3D, train_y, K, var_explained, 'training', 30, 45)

    ml.plotDataIn2D(lda_test_2D, test_y, K, x_label, y_label, 'LDA', 'test')
    ml.plotDataIn3D(lda_test_3D, test_y, K, x_label, y_label, z_label, 'LDA', 'test', 20, 15)

    lplots.plot2DwithDecisionsLDA(W_2D, lda_test_2D, test_y, K, var_explained, 'test')
    lplots.plot3DwithDecisionsLDA(W_3D, lda_test_3D, test_y, K, var_explained, 'test', 20, 15)


# Call for least squares classifier training on LDA data
def train_and_test_lda(train_set, test_set, K, d):

    # Separate training and test data into attributes and class labels
    train_x, train_y, test_x, test_y, train_y_binary = ml.splitIntoFeaturesAndLabels(test_set, train_set, K, d)

    # Train the least squares classifier model on the training data data
    W = performLeastSquareClassifier(train_x, train_y_binary)

    # Compute model performance on training data (Average training accuracy)
    ave_acc_train = predictLabels(train_x, train_y, W, K)
    # Compute model performance on test data (Average testing accuracy)
    ave_acc_test = predictLabels(test_x, test_y, W, K)
    print("Average training accuracy (least squares classifier): {}".format(ave_acc_train))
    print("Average test accuracy (least squares classifier): {}".format(ave_acc_test))

    # Perform LDA on the training data and use the same projection to compute dimensionality-reduced test data
    lda_train, var_explained, eigVectors = reduceDimensionsUsingLDA(train_x, train_y, K, d)
    lda_test = eigVectors.dot(test_x.transpose())

    lda_ave_acc_train = []
    lda_ave_acc_test = []
    x_ticks = []
    D = K + 2
    for i in range(1, D):
        # Extract a set number of dimensions from the LDA training and test data
        lda_train_sub = lda_train[0:i, :].transpose()
        lda_test_sub = lda_test[0:i, :].transpose()

        # Add intercept to the LDA training subset and train the least squares classifier
        lda_W = performLeastSquareClassifier(lda_train_sub, train_y_binary)

        # Test the model weights on the LDA training subset and collect Average training accuracies
        lda_ave_acc_train.append(predictLabels(lda_train_sub, train_y, lda_W, K))
        # Test the model weights on the LDA test subset and collect Average test accuracies
        lda_ave_acc_test.append(predictLabels(lda_test_sub, test_y, lda_W, K))
        x_ticks.append("d = {}\nve = {:0.2f}%".format(i, var_explained[i-1]))

    print("Average accuracy (training data on LDA least squares classifier): {}".format(lda_ave_acc_train))
    print("Average accuracy (test data on LDA least squares classifier): {}".format(lda_ave_acc_test))

    # Plot model performances and LDA projections
    plotPerformanceOverDimensions(lda_ave_acc_test, x_ticks, "Least squares classifier performance on LDA dimensionality-reduced data\n", D)
    initiateCreationOfLDAPlots(lda_train, lda_test, train_y_binary, train_y, test_y, var_explained, K)


# Average training accuracy (least squares classifier): 0.95148
# Average test accuracy (least squares classifier): 0.91744
# Average accuracy (training data on LDA least squares classifier): [0.7595200000000001, 0.85112, 0.9028799999999999, 0.9514800000000001, 0.9514800000000001, 0.9514800000000001]
# Average accuracy (test data on LDA least squares classifier): [0.7456, 0.83568, 0.8881599999999998, 0.9174399999999999, 0.9174399999999999, 0.9174399999999999]
