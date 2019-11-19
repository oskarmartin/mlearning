import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import random


# reshape til 28*28
# plt.imshow

scatter_shapes = ['o', 'v', 's', '*', 'x']
colors = ['gray', 'tab:purple', 'tab:blue', 'orange', 'pink']

pixel_range = 255.0
number_of_classes = 5
number_of_dimensions = 28 * 28

data_test = np.load("data/fashion_test.npy")

test_x = np.array(data_test[:, 0:number_of_dimensions]) / pixel_range
test_y = np.array(data_test[:, number_of_dimensions])

test_x_tilde = test_x - np.mean(test_x)

# Compute covariance matrix, eigenvalues and eigenvectors (= PCs)
cov = (test_x_tilde.transpose().dot(test_x_tilde))/number_of_dimensions
eigenValues, eigenVectors = np.linalg.eig(cov)
desc_eig = np.argsort(eigenValues)[::-1]


pc = (eigenVectors.transpose().dot(test_x_tilde.transpose()))

print(pc.shape)

print(pc[0])
print(test_y)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
for i in range(number_of_classes):

    class_i_x = np.take(pc[0], np.where(test_y == i))
    class_i_y = np.take(pc[1], np.where(test_y == i))
    class_i_z = np.take(pc[2], np.where(test_y == i))

    plt.title('Data projection onto three principle components\n\n\n\n\n')
    plt.xlabel(u'\nPC 1: \u03C3$^2$ = {:0.2f}'.format(np.var(pc[0])))
    plt.ylabel(u'\nPC 2: \u03C3$^2$ = {:0.2f}'.format(np.var(pc[1])))
    ax.set_zlabel(u'\nPC 3: \u03C3$^2$ = {:0.2f}'.format(np.var(pc[2])))
    ax.scatter3D(class_i_x, class_i_y, class_i_z, c=colors[i], alpha=0.3)
    ax.view_init(azim=150)

fig.savefig('plots/3Dpca.png')
plt.show()

fig = plt.figure()
for i in range(number_of_classes):

    class_i_x = np.take(pc[0], np.where(test_y == i))
    class_i_y = np.take(pc[1], np.where(test_y == i))

    plt.scatter(class_i_x, class_i_y, c=colors[i], alpha=0.3)
    plt.title('Data projection onto two principle components\n')
    plt.xlabel(u'PC 1: \u03C3$^2$ = {:0.2f}'.format(np.var(pc[0])))
    plt.ylabel(u'PC 2: \u03C3$^2$ = {:0.2f}'.format(np.var(pc[1])))

fig.savefig('plots/2Dpca.png')
plt.show()

