import numpy as np
import matplotlib.pyplot as plt
import random
import ml_tools as ml

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


train_x, train_y, test_x, test_y, train_y_binary = ml.splitIntoFeaturesAndLabels(data_test, data_train, number_of_classes, number_of_dimensions)


def expandData(data_x, d):

    data_x_2 = np.zeros(data_x.shape)

    image_width = int(np.sqrt(d))

    for i in range(data_x.shape[0]):
        image = data_x[i, :].reshape(28, 28)

        flipped = np.zeros((image_width, image_width))

        for j in range(image_width):
            flipped[j, :] = image[j, :][::-1]

        data_x_2[i] = np.array(flipped).flatten()

    extended_data_x = np.vstack((data_x, data_x_2))

    print(extended_data_x.shape)

    return extended_data_x

expandData(train_x, number_of_dimensions)