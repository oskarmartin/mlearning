import numpy as np
import matplotlib.pyplot as plt
import math

colors = ["orange", "purple", "gray", "red", "blue"]

data_train = np.load("data/fashion_train.npy")
data_test = np.load("data/fashion_test.npy")

no_of_classes = 5
number_of_dimensions = 28 * 28
train_size = len(data_train)
test_size = len(data_test)

train_x = []
train_y = []
test_x = []
test_y = []

for i in range(train_size):
    train_x.append(data_train[i][0:number_of_dimensions])
    train_y.append(data_train[i][number_of_dimensions])

for i in range(test_size):
    test_x.append(data_test[i][0:number_of_dimensions])
    test_y.append(data_test[i][number_of_dimensions])

# is this correct??
def normalize(m):

    for i in range(len(m)):
        m[i] = (m[i] - np.mean(m[i])) / np.std(m[i])

    return m


# print(train_x[0])
# print(normalize(train_x)[0])


train_x = np.array(normalize(train_x))
train_y = np.array(train_y)
test_x = np.array(normalize(test_x))
test_y = np.array(test_y)

# print(len(train_x))
# print(len(train_y))
# print(len(test_x))
# print(len(test_y))


for i in range(12):
    plt.scatter(test_x[i][287], test_x[i][303], color=colors[test_y[i]])

plt.show()


def logistic_discrimination(x, r, K, d, iterations, eta):

    print("*** Training logistic discrimination model ***")

    w = np.random.uniform(-0.01, 0.01, (K, d))
    y = np.zeros((len(x), K))
    E = np.zeros(iterations)
    abs_Wj = np.zeros((iterations, K))

    for it in range(iterations):
        print("Iteration: {}".format(it))
        grad_w = np.zeros((K, d))

        for t in range(len(x)):
            o = np.zeros(K)
            for i in range(K):
                for j in range(d):
                    # print(w[i][j])
                    o[i] += w[i][j] * x[t][j]

            for i in range(K):
                # print("iteration: {}".format(it))
                # print("image: {}".format(t))
                # print("o[i]: {}".format(o[i]))
                # print("exp(o[i]): {}".format(np.exp(o[i])))
                # print("np.sum(np.exp(o)): {}".format(np.sum(np.exp(o))))
                y[t][i] = np.exp(o[i])/np.sum(np.exp(o))
                # print(y[t][i])

            for i in range(K):
                for j in range(d):
                    grad_w[i][j] += (r[t][i] - y[t][i]) * x[t][j]

        for i in range(K):
            for j in range(d):
                w[i][j] += eta * grad_w[i][j]
                # print ("w[i][j]: {}".format(w[i][j]))

        for i in range(K):
            abs_Wj[it][i] = sum(abs(w[i][0:d-1]))

        # print ("it 1: {}".format(it))
        for t in range(len(x)):
            for i in range(K):
                # print ("it 2: {}".format(it))
                # print ("r[t][i]: {}".format(r[t][i]))
                # print ("math.log((y[t][i]), 10): {}".format(math.log((y[t][i]), 10)))
                E[it] -= r[t][i] * math.log((y[t][i]), 10)

    return w, E, abs_Wj

def labels_to_matrix(y, K):

    t = len(y)
    y_matrix = np.zeros((t, K))

    for i in range(t):
        y_matrix[i][y[i]] = 1

    return y_matrix

x = train_x[0:1000]
r = labels_to_matrix(train_y[0:1000], no_of_classes)
print(x.shape)
weights, cross_entropy, sigmoid_slope = logistic_discrimination(x, r, no_of_classes, number_of_dimensions, 100, 0.00006)
#weights, cross_entropy = logistic_discrimination(train_x, labels_to_matrix(train_y, no_of_classes), no_of_classes, number_of_dimensions, 10, 0.008)

plt.plot(np.arange(len(cross_entropy)), cross_entropy)
plt.show()

print(cross_entropy)
print(sigmoid_slope)
print(sigmoid_slope.shape)

for i in range(sigmoid_slope.shape[1]):
    plt.plot(np.arange(sigmoid_slope.shape[0]), sigmoid_slope[:, i])

plt.show()