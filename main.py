import logistic_regression as lg
import pca
import random
import numpy as np

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

# Perform model training, testing and plot evaluation
lg.crossValidation(data_train, data_test, 5, number_of_classes, number_of_dimensions, iterations_range, step_size_range, pixel_range)
# Plot dimensionality reduced data
pca.performPCA(data_test, number_of_classes, number_of_dimensions, 150)
# Perform linear discriminant analysis

# reshape til 28*28
# plt.imshow
# v = np.random.randint(0, 10000, 1000)