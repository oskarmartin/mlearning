import sys

import numpy as np
import ml_tools as ml
import pca
import random
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d

# Known data information
number_of_classes = 5
number_of_dimensions = 28 * 28
pixel_range = 255.0

# Load data
data_train = np.load("data/fashion_train.npy")
data_test = np.load("data/fashion_test.npy")

# Shuffle data
random.shuffle(data_train)
random.shuffle(data_test)

# Parameter settings for computing varying models
iterations_range = np.array([250, 300, 450, 500])
step_size_range = np.array([0.0001, 0.000015, 0.00001, 0.0000015])

scatter_shapes = ['o', 'v', 's', '*', 'x']
colors = ['gray', 'tab:purple', 'tab:blue', 'orange', 'pink']


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

    # Project data onto the eigenvectors of J
    Z = eigVectors_sorted.dot(X.transpose())

    return Z, var_explained


# Compute decision boundary weights of input data using Least Square Classifier
def performLeastSquareClassifier(X, T):

    d = X.shape[1]
    X_extended = np.ones((X.shape[0], d + 1))
    X_extended[:, 0:d] = X

    W = np.linalg.inv((X.transpose()).dot(X)).dot((X.transpose()).dot(T))

    return W

# Compare predicted labels to true labels and output Average Accuracy
def predictLabels(test_data_x, true_labels, trained_weights, K):

    # d = test_data.shape[1]
    # test_data_extended = np.ones((test_data.shape[0], d + 1))
    # test_data_extended[:, 0:d] = test_data

    probability_y = test_data_x.dot(trained_weights)
    y = np.argmax(probability_y, axis=1)
    ave_acc = ml.averageAccuracy(true_labels, y, K)

    return ave_acc

# Plot scatter of input data in 3 dimensions with its decision boundaries defined by input weights
def plot3DwithDecisionsLDA(W, data_x, data_y, K, var_explained, title):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    boundary_x = np.linspace(np.min(data_x[0, :]), np.max(data_x[0, :]), 30)
    boundary_y = np.linspace(np.min(data_x[1, :]), np.max(data_x[1, :]), 30)
    boundary_x, boundary_y = np.meshgrid(boundary_x, boundary_y)

    for i in range(K):

        class_i_x = np.take(data_x[0], np.where(data_y == i))
        class_i_y = np.take(data_x[1], np.where(data_y == i))
        class_i_z = np.take(data_x[2], np.where(data_y == i))

        ax.scatter3D(class_i_x, class_i_y, class_i_z, c=colors[i], alpha=0.3)

        boundary_z = W[0, i] * boundary_x + W[1, i] * boundary_y + W[2, i]
        ax.contour3D(boundary_x, boundary_y, boundary_z, 60, cmap='binary')

    plt.title(title)
    plt.xlabel('\n\n\nve = {:0.2f}%'.format(var_explained[0]))
    plt.ylabel('\n\n\nve = {:0.2f}%'.format(var_explained[1]))
    ax.set_zlabel('\n\n\nve = {:0.2f}%'.format(var_explained[2]))
    ax.view_init(15, 173)
    fig.savefig('plots/3Dlda_with_Hyperplanes.png')
    plt.show()

# Plot scatter of input data in 2 dimensions with its decision boundaries defined by input weights
def plot2DwithDecisionsLDA(W, data_x, data_y, K, var_explained, title):

    fig = plt.figure()

    boundary_x = np.linspace(np.min(data_x[0, :]), np.max(data_x[0, :]), 30)

    for i in range(K):

        class_i_x = np.take(data_x[0], np.where(data_y == i))
        class_i_y = np.take(data_x[1], np.where(data_y == i))
        plt.scatter(class_i_x, class_i_y, c=colors[i], alpha=0.3)

        boundary_y = W[0, i] * boundary_x + W[1, i]
        plt.plot(boundary_x, boundary_y, c = 'gray')

    plt.title(title)
    plt.xlabel('\nve = {:0.2f}%'.format(var_explained[0]))
    plt.ylabel('\nve = {:0.2f}%'.format(var_explained[1]))
    fig.savefig('plots/2Dlda_with_Hyperplanes.png')
    plt.show()


def discriminate(train_set, test_set, K, d):

    train_x, train_y, test_x, test_y, train_y_binary = ml.splitIntoFeaturesAndLabels(test_set, train_set, K, d)

    W = performLeastSquareClassifier(train_x, train_y_binary)

    ave_acc = predictLabels(test_x, test_y, W, K)
    print("Least square classifier average test accuracy: {}".format(ave_acc))
    # ---------------------------------------------------------------------- #

    lda_train_x, var_explained = reduceDimensionsUsingLDA(train_x, train_y, K, d)
    lda_test_x, _ = reduceDimensionsUsingLDA(test_x, test_y, K, d)
    ave_acc = []
    x_ticks = []

    for i in range(1, K+1):
        print(i)
        lda_train_x_sub = lda_train_x[0:i, :].transpose()
        lda_test_x_sub = lda_test_x[0:i, :].transpose()

        W = performLeastSquareClassifier(lda_train_x, train_y_binary)

        ave_acc.append(predictLabels(lda_test_x_sub, test_y, W, K))
        x_ticks.append("d = {}\nve = {:0.2f}%".format(i, var_explained[i]))

    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(ave_acc)), ave_acc)
    plt.xticks(np.arange(len(x_ticks)), x_ticks)
    plt.show()

    print(ave_acc)

    lda_train_x_2D = lda_train_x[0:2, :]
    lda_train_x_3D = lda_train_x[0:3, :]

    W_2D = performLeastSquareClassifier(lda_train_x_2D.transpose(), train_y_binary)
    W_3D = performLeastSquareClassifier(lda_train_x_3D.transpose(), train_y_binary)

    x_label = 've = {:0.2f}%'.format(var_explained[0])
    y_label = 've = {:0.2f}%'.format(var_explained[1])
    z_label = 've = {:0.2f}%'.format(var_explained[2])

    pca.pcaPlot2d(lda_train_x_2D, train_y, K, x_label, y_label, 'PCA training 2D data projection\n')
    pca.pcaPlot3d(lda_train_x_3D, train_y, K, x_label, y_label, z_label, 'PCA 3D training data projection\n', 150)

    plot2DwithDecisionsLDA(W_2D, lda_train_x_2D, train_y, K, var_explained, 'LDA 2D training data projection\nwith least square classifier decision boundaries')
    plot3DwithDecisionsLDA(W_3D, lda_train_x_3D, train_y, K, var_explained, 'LDA 3D training data projection\nwith least square classifier decision boundaries\n')



discriminate(data_train, data_test, number_of_classes, number_of_dimensions)



