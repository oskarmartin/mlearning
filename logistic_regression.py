import numpy as np
import matplotlib.pyplot as plt
import math

# reshape til 28*28
# plt.imshow

colors = ["orange", "purple", "gray", "red", "blue"]
pixel_range = 255.0

data_train = np.load("data/fashion_train.npy")
data_test = np.load("data/fashion_test.npy")

number_of_classes = 5
number_of_dimensions = 28 * 28
train_size = len(data_train)
test_size = len(data_test)

train_x = np.array(data_train[:, 0:number_of_dimensions])/pixel_range
train_y = np.array(data_train[:, number_of_dimensions])
test_x = np.array(data_test[:, 0:number_of_dimensions])/pixel_range
test_y = np.array(data_test[:, number_of_dimensions])

train_y_binary = np.zeros((train_size, number_of_classes))
train_y_binary[np.arange(train_size), train_y] = 1


def logistic_regression(x, r, K, d, iterations, eta):

    print("*** Training logistic discrimination model ***")

    train_set_size = len(x)
    w = np.random.uniform(-0.01, 0.01, (K, d))
    y = np.zeros((train_set_size, K))
    cross_e = np.zeros(iterations)
    acc = np.zeros(iterations)

    for it in range(iterations):
        print("Iteration: {}".format(it))
        grad_w = np.zeros((K, d))

        for t in range(train_set_size):
            o = (w * x[t]).sum(axis=1)
            y[t] = np.exp(o) / np.sum(np.exp(o))

            if np.argmax(y[t]) == np.argmax(r[t]):
                acc[it] += 1.0/float(train_set_size)

            for i in range(K):
                grad_w[i] += (r[t][i] - y[t][i]) * x[t]

        w += eta * grad_w

        correct_class_indices = np.argmax(r, axis=1)
        prob_correct_class = np.choose(correct_class_indices, y.T)
        cross_e[it] = -sum(np.log(prob_correct_class)/float(train_set_size))

    return w, cross_e, acc


v = np.random.randint(0, 10000, 1000)

train_x_subset = train_x[v]
train_y_subset = train_y[v]
train_y_binary_subset = train_y_binary[v]

weights, cross_entropy, correct_classification = logistic_regression(train_x_subset, train_y_binary_subset, number_of_classes, number_of_dimensions, 50, 0.0001)

plt.plot(np.arange(len(correct_classification)), correct_classification, color='orange')
plt.show()

plt.plot(np.arange(len(cross_entropy)), cross_entropy)
plt.show()