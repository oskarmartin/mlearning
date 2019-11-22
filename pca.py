import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

scatter_shapes = ['o', 'v', 's', '*', 'x']
colors = ['gray', 'tab:purple', 'tab:blue', 'orange', 'pink']

def dimensionalityReduction(data_x, d):

    # Normalize data
    data_x_tilde = (data_x - np.mean(data_x)) / np.std(data_x)

    # Compute covariance matrix, eigenvalues and eigenvectors (= principle components (PC))
    cov = (data_x_tilde.transpose().dot(data_x_tilde)) / d
    eigValues, eigVectors = np.linalg.eig(cov)
    eigVectors_sorted = eigVectors[np.argsort(eigValues)[::-1]]

    # Project data onto the 2 and 3 first principle components respectively
    pro_2d = (eigVectors_sorted[:, 0:2].transpose().dot(data_x_tilde.transpose()))
    pro_3d = (eigVectors_sorted[:, 0:3].transpose().dot(data_x_tilde.transpose()))

    return pro_2d, pro_3d

def pcaPlot3d(pro_3d, data_y, K):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(K):

        class_i_x = np.take(pro_3d[0], np.where(data_y == i))
        class_i_y = np.take(pro_3d[1], np.where(data_y == i))
        class_i_z = np.take(pro_3d[2], np.where(data_y == i))

        ax.scatter3D(class_i_x, class_i_y, class_i_z, c=colors[i], alpha=0.3)

    plt.title('Data projection onto three principle components\n')
    plt.xlabel(u'\nPC 1: \u03C3$^2$ = {:0.2f}'.format(np.var(pro_3d[0, :])))
    plt.ylabel(u'\nPC 2: \u03C3$^2$ = {:0.2f}'.format(np.var(pro_3d[1, :])))
    ax.set_zlabel(u'PC 3: \u03C3$^2$ = {:0.2f}'.format(np.var(pro_3d[2, :])))
    ax.view_init(azim=150)
    fig.savefig('plots/3Dpca.png')
    plt.show()

def pcaPlot2d(pro_2d, data_y, K):

    fig = plt.figure()
    for i in range(K):

        class_i_x = np.take(pro_2d[0], np.where(data_y == i))
        class_i_y = np.take(pro_2d[1], np.where(data_y == i))

        plt.scatter(class_i_x, class_i_y, c=colors[i], alpha=0.3)

    plt.title('Data projection onto two principle components\n')
    plt.xlabel(u'PC 1: \u03C3$^2$ = {:0.2f}'.format(np.var(pro_2d[0, :])))
    plt.ylabel(u'PC 2: \u03C3$^2$ = {:0.2f}'.format(np.var(pro_2d[1, :])))
    fig.savefig('plots/2Dpca.png')
    plt.show()

def performPCA(data, K, d):

    data_x = np.array(data[:, 0:d])
    data_y = np.array(data[:, d])
    projection_2D, projection_3D = dimensionalityReduction(data_x, d)
    pcaPlot2d(projection_2D, data_y, K)
    pcaPlot3d(projection_3D, data_y, K)