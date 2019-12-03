import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

scatter_shapes = ['o', 'v', 's', '*', 'x']
colors = ['gray', 'tab:purple', 'tab:blue', 'orange', 'pink']

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


def pcaPlot3d(pro_3d, data_y, K, x_label, y_label, z_label, title, angle):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(K):

        class_i_x = np.take(pro_3d[0], np.where(data_y == i))
        class_i_y = np.take(pro_3d[1], np.where(data_y == i))
        class_i_z = np.take(pro_3d[2], np.where(data_y == i))

        ax.scatter3D(class_i_x, class_i_y, class_i_z, c=colors[i], alpha=0.3)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.view_init(azim=angle)
    fig.savefig('plots/3Dpca.png')
    plt.show()

    return

def pcaPlot2d(pro_2d, data_y, K, x_label, y_label, title):

    fig = plt.figure()
    for i in range(K):

        class_i_x = np.take(pro_2d[0], np.where(data_y == i))
        class_i_y = np.take(pro_2d[1], np.where(data_y == i))

        plt.scatter(class_i_x, class_i_y, c=colors[i], alpha=0.3)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    fig.savefig('plots/2Dpca.png')
    plt.show()

def performPCA(data, K, d, angle):

    data_x = np.array(data[:, 0:d])
    data_y = np.array(data[:, d])
    projection_2D, projection_3D, variance_explained = reduceDimensionsUsingPCA(data_x, d)

    x_label = 'PC 1: ve = {:0.2f}%'.format(variance_explained[0])
    y_label = 'PC 2: ve = {:0.2f}%'.format(variance_explained[1])
    z_label = 'PC 3: ve = {:0.2f}%'.format(variance_explained[2])

    pcaPlot2d(projection_2D, data_y, K, x_label, y_label, 'PCA data 2D projection\n')
    pcaPlot3d(projection_3D, data_y, K, x_label, y_label, z_label, 'PCA data 3D projection\n', angle)