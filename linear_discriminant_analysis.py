import numpy as np
import data_processing as dp
import pca
import random
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

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


def linear_discriminant_analysis(x, y, K, d):

    data_dim_mean = (np.mean(x, axis=0)).reshape(d, 1)

    class_size = np.zeros(K).astype(int)
    class_dim_mean = []
    S_W = np.zeros((d, d))
    for i in range(K):
        class_i = x[y == i]
        class_size[i] = class_i.shape[0]
        class_dim_mean.append(np.mean(class_i, axis=0))
        m_i = class_dim_mean[i].reshape(d, 1)

        # Within-class scatter matrix S_i
        S_i = np.zeros((d, d))
        for n in range(class_size[i]):
            x_t = class_i[n].reshape(d, 1)
            S_i += (x_t - m_i).dot((x_t - m_i).transpose())

        # Total within-class scatter matrix S_W
        S_W += S_i

    # Between class scatter matrix S_B
    S_B = np.zeros((d, d))
    for i in range(K):
        m_i = class_dim_mean[i].reshape(d, 1)
        S_B += class_size[i] * (m_i - data_dim_mean).dot((m_i - data_dim_mean).transpose())


    SW_SB = np.linalg.inv(S_W).dot(S_B)

    print("SW_SB shape: {}".format(SW_SB.shape))

    eigValues, eigVectors = np.linalg.eig(SW_SB)
    eigVectors_sorted = eigVectors[np.argsort(eigValues)][::-1]

    Z = np.real((x.dot(eigVectors_sorted)))
    pro_2d = Z[:, 0:2].transpose()
    pro_3d = Z[:, 0:3].transpose()

    print("Z shape: {}".format(Z.shape))
    print("eigVectors_sorted shape: {}".format(eigVectors_sorted.shape))
    print("pro_2d shape: {}".format(pro_2d.shape))
    print("pro_3d shape: {}".format(pro_3d.shape))

    pca.pcaPlot2d(pro_2d, y, K)
    pca.pcaPlot3d(pro_3d, y, K, 160)

def discriminate(train_set, test_set, K, d, element_range):
    train_x, train_y, test_x, test_y, train_y_binary = dp.splitIntoAttAndClass(test_set, train_set, K, d, element_range)
    linear_discriminant_analysis(test_x, test_y, K, d)


discriminate(data_train, data_test, number_of_classes, number_of_dimensions, pixel_range)

a = np.array([3, 4, 5, 2, 1])
c = np.array([3, 4, 5, 2, 1])
b = np.argsort(a)
print(a[b][::-1])