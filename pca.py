import numpy as np
import ml_tools as ml

# Compute data reduced to 2 and 3 dimensions using PCA
def reduceDimensionsUsingPCA(data_x, d):

    # Normalize data
    data_x_tilde = (data_x - np.mean(data_x)) / np.std(data_x)

    # Compute covariance matrix, eigenvalues and eigenvectors (= principle components (PC))
    cov = (data_x_tilde.transpose().dot(data_x_tilde)) / d
    eigValues, eigVectors = np.linalg.eig(cov)
    eigVectors_sorted = eigVectors[np.argsort(eigValues)[::-1]]
    eigValues_sorted = np.sort(eigValues)[::-1]
    var_explained = (eigValues_sorted / np.sum(eigValues_sorted)) * 100


    # Project data onto the 2 and 3 first principle components respectively
    data_x_2d = (eigVectors_sorted[:, 0:2].transpose().dot(data_x_tilde.transpose()))
    data_x_3d = (eigVectors_sorted[:, 0:3].transpose().dot(data_x_tilde.transpose()))

    return data_x_2d, data_x_3d, var_explained


# Call for PCA dimensionality reduction and plotting
def computePCAPlots(data, K, d):

    data_x = np.array(data[:, 0:d])
    data_y = np.array(data[:, d])
    projection_2D, projection_3D, variance_explained = reduceDimensionsUsingPCA(data_x, d)

    x_label = 'PC 1: ve = {:0.2f}%'.format(variance_explained[0])
    y_label = 'PC 2: ve = {:0.2f}%'.format(variance_explained[1])
    z_label = 'PC 3: ve = {:0.2f}%'.format(variance_explained[2])

    ml.plotDataIn2D(projection_2D, data_y, K, x_label, y_label, "PCA", "")
    ml.plotDataIn3D(projection_3D, data_y, K, x_label, y_label, z_label, "PCA", "", 30, 125)