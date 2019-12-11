import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


scatter_shapes = ['o', 'v', 's', '*', 'x']
colors = ['gray', 'tab:purple', 'tab:blue', 'orange', 'pink']


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
    # train_x_std[train_x_std == 0] = 0.00000001         # kan jeg tillade mig dette?
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
    data_extend = np.zeros((extended_data_x.shape[0], d+1))
    data_extend[:, 0:d] = extended_data_x
    data_extend[:, d] = extended_data_y
    data_extend = data_extend.astype(int)

    return data_extend


# Compute confusion matrix
def printConfusionMatrix(true_classes, pred_classes, K):

    confMatrixInt = np.zeros((K, K))
    confMatrixString = [[ "" for i in range(K)] for j in range(K)]
    for i in range(len(true_classes)):
        confMatrixInt[true_classes[i]][pred_classes[i]] += 1

    for i in range(K):
        for j in range(K):
            class_size = np.count_nonzero(true_classes == i)
            if class_size != 0:
                confMatrixInt[i][j] = (confMatrixInt[i][j] / class_size) * 100
            confMatrixString[i][j] = str(round(confMatrixInt[i][j], 1)) + "%"

    print(np.array(confMatrixString))

    return confMatrixString


# Plot data in 2D and color according to class labels
def plotDataIn2D(pro_2d, data_y, K, x_label, y_label, dim_reducer, set_type):

    fig = plt.figure()
    for i in range(K):

        class_i_x = np.take(pro_2d[0], np.where(data_y == i))
        class_i_y = np.take(pro_2d[1], np.where(data_y == i))

        plt.scatter(class_i_x, class_i_y, c=colors[i], alpha=0.3)

    plt.title("{} 2D {} data projection \n".format(dim_reducer, set_type))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    fig.savefig('projectionPlots/2D_{}_plot_{}.png'.format(set_type, dim_reducer))
    plt.show()


# Plot data in 3D and color according to class labels
def plotDataIn3D(pro_3d, data_y, K, x_label, y_label, z_label, dim_reducer, set_type, angle_x, angle_y):

    # Set ax properties
    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection='3d')
    if set_type == 'test':
        plt.xlim(-2, 3)
        plt.ylim(-2, 2)
        ax.set_zlim(-2, 2)

    # Give each class a unique scatter color
    for i in range(K):
        class_i_x = np.take(pro_3d[0], np.where(data_y == i))
        class_i_y = np.take(pro_3d[1], np.where(data_y == i))
        class_i_z = np.take(pro_3d[2], np.where(data_y == i))
        ax.scatter3D(class_i_x, class_i_y, class_i_z, c=colors[i], alpha=0.3)

    # Set plot labels
    plt.title("{} 3D {} data projection \n".format(dim_reducer, set_type))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.view_init(angle_x, angle_y)

    # Set file saving name
    if set_type == 'training' and pro_3d.shape[1] > 10000:
        fig.savefig('projectionPlots/3D_{}_plot_{}_extended.png'.format(set_type, dim_reducer))
    else:
        fig.savefig('projectionPlots/3D_{}_plot_{}.png'.format(set_type, dim_reducer))
    plt.show()

    return


