import numpy as np
import matplotlib.pyplot as plt
import random

# Load data
data_train = np.load("data/fashion_train.npy")
data_test = np.load("data/fashion_test.npy")

# Shuffle data
random.shuffle(data_train)
random.shuffle(data_test)

train_x, train_y, test_x, test_y, train_y_binary = dp.splitIntoFeaturesAndLabels(test_set, train_set, K, d, element_range)


plt.imshow