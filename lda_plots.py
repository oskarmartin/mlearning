import numpy as np
from matplotlib import pyplot as plt

scatter_shapes = ['o', 'v', 's', '*', 'x']
colors = ['gray', 'tab:purple', 'tab:blue', 'orange', 'pink']


# Plot scatter of input data in 2 dimensions with its decision boundaries defined by input weights
def plot2DwithDecisionsLDA(W, data_x, data_y, K, var_explained, set_type):

    # Set axis properties
    fig = plt.figure()
    boundary_x = np.linspace(np.min(data_x[0, :]), np.max(data_x[0, :]), 30)

    # Give each class a unique scatter color
    for i in range(K):
        class_i_x = np.take(data_x[0], np.where(data_y == i))
        class_i_y = np.take(data_x[1], np.where(data_y == i))
        plt.scatter(class_i_x, class_i_y, c=colors[i], alpha=0.3)

        boundary_y = W[0, i] * boundary_x + W[1, i]
        plt.plot(boundary_x, boundary_y, c = 'gray')

    # Set plot labels
    plt.title("LDA 2D {} data projection\nwith least squares classifier decision boundaries".format(set_type))
    plt.xlabel('\nve = {:0.2f}%'.format(var_explained[0]))
    plt.ylabel('\nve = {:0.2f}%'.format(var_explained[1]))
    fig.savefig('projectionPlots/2D_{}_plot_LDA_w_boundaries.png'.format(set_type))
    plt.show()


# Plot scatter of input data in 3 dimensions with decision boundaries defined by input weights
def plot3DwithDecisionsLDA(W, data_x, data_y, K, var_explained, set_type, angle_x, angle_y):

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection='3d')

    # Set axis properties
    if set_type == 'test':
        plt.xlim(-2, 3)
        plt.ylim(-2, 2)
        ax.set_zlim(-2, 2)
        boundary_x = np.linspace(-2, 3, 30)
        boundary_y = np.linspace(-2, 2, 30)
    else:
        boundary_x = np.linspace(np.min(data_x[0, :]), np.max(data_x[0, :]), 30)
        boundary_y = np.linspace(np.min(data_x[1, :]), np.max(data_x[1, :]), 30)
    boundary_x, boundary_y = np.meshgrid(boundary_x, boundary_y)

    # Give each class a unique scatter color
    for i in range(K):
        class_i_x = np.take(data_x[0], np.where(data_y == i))
        class_i_y = np.take(data_x[1], np.where(data_y == i))
        class_i_z = np.take(data_x[2], np.where(data_y == i))

        boundary_z = W[2, i] * boundary_x + W[1, i] * boundary_y + W[0, i]
        ax.scatter3D(class_i_x, class_i_y, class_i_z, c=colors[i], alpha=0.3)
        ax.contour3D(boundary_x, boundary_y, boundary_z, 60, cmap='binary')

    # Set plot labels
    if angle_x < 0:
        plt.title("LDA 3D {} data projection\nwith Least Squares classifier decision boundaries\n\n\n".format(set_type))
    else:
        plt.title("LDA 3D {} data projection\nwith Least Squares classifier decision boundaries\n".format(set_type))

    plt.xlabel('\nve = {:0.2f}%'.format(var_explained[0]))
    plt.ylabel('\nve = {:0.2f}%'.format(var_explained[1]))
    ax.set_zlabel('\nve = {:0.2f}%'.format(var_explained[2]))
    ax.view_init(angle_x, angle_y)

    # Set file saving name
    if set_type == 'training' and data_x.shape[1] > 10000:
        fig.savefig('projectionPlots/3D_{}_plot_LDA_w_boundaries_extended.png'.format(set_type))
    else:
        fig.savefig('projectionPlots/3D_{}_plot_LDA_w_boundaries.png'.format(set_type))
    plt.show()
