import logistic_regression as lg
import pca
import random
import numpy as np
import ml_tools as ml

# Known data information
number_of_classes = 5
number_of_dimensions = 28 * 28

# Load data
data_train = np.load("data/fashion_train.npy")
data_test = np.load("data/fashion_test.npy")

# Shuffle data
random.shuffle(data_train)
random.shuffle(data_test)

# Extend training data by its flipped elements
data_train_extend = ml.extendData(data_train, number_of_dimensions)

# Parameter settings for computing varying models
# iterations_range = np.array([10, 11, 12, 13])
iterations_range = np.array([200, 300, 400, 500])
step_size_range = np.array([0.000015, 0.00001, 0.0000015, 0.000001])
number_of_CV_folds = 5

# Perform model training, testing and plot evaluation
lg.crossValidation(data_train_extend, data_test, number_of_CV_folds, number_of_classes, number_of_dimensions, iterations_range, step_size_range)
# Plot dimensionality reduced data
pca.performPCA(data_test, number_of_classes, number_of_dimensions, 170)
# Perform linear discriminant analysis

# reshape til 28*28
# plt.imshow
# v = np.random.randint(0, 10000, 1000)