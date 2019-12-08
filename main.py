import logistic_regression as lg
import pca
import random
import numpy as np
import ml_tools as ml
import lda

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

# Perform logistic regression
lg.log_reg_cross_validation(data_train_extend, data_test, number_of_CV_folds, number_of_classes, number_of_dimensions, iterations_range, step_size_range)
# Perform linear discriminant analysis
# lda.train_and_test_lda(data_train_extend, data_test, number_of_classes, number_of_dimensions)
# Plot PCA dimensionality-reduced training and test data
# pca.computePCAPlots(data_train, number_of_classes, number_of_dimensions, 170)
# pca.computePCAPlots(data_test, number_of_classes, number_of_dimensions, 170)

