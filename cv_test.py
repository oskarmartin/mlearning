import numpy as np
import matplotlib.pyplot as plt
import math

train_size = 10
number_of_classes = 5
train_y = np.array([2, 3, 1, 0, 0, 4, 2, 0, 4, 1])


train_y_binary = np.zeros((train_size, number_of_classes))

print(train_y_binary)

train_y_binary[np.arange(train_size), train_y] = 1

print(train_y_binary)