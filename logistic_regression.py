
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# reshape til 28*28
# plt.imshow
# v = np.random.randint(0, 10000, 1000)

number_of_classes = 5
number_of_dimensions = 28 * 28
pixel_range = 255.0

data_train = np.load("data/fashion_train.npy")
data_test = np.load("data/fashion_test.npy")


def logistic_regression(x, r, K, d, iterations, eta):

    print("\tTraining initialized")

    train_set_size = len(x)
    w = np.random.uniform(-0.01, 0.01, (K, d))
    y = np.zeros((train_set_size, K))
    cross_e = np.zeros(iterations)
    acc = np.zeros(iterations)

    for it in range(iterations):
        print("\r\t\tRunning iteration {} of {}".format(it+1, iterations)),
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


def plotAABarPlot(aa_array):
    folds = len(aa_array)
    plt.title("Testing logistic regression model")
    plt.bar(np.arange(folds), aa_array * 100, 0.35)
    for f in range(folds):
        plt.annotate("{}%".format(aa_array[f] * 100), (f - 0.2, aa_array[f] * 100 + 1))
    plt.ylabel("Average model accuracy (%)")
    plt.xlabel("CV fold")
    plt.show()

def plotCCGraph(cc_array, fold):
    plt.title("Cross-validation fold: {}\nTraining logistic regression model".format(fold))
    plt.plot(np.arange(len(cc_array)), cc_array, color='orange')
    plt.ylabel("Correct classifications (%)")
    plt.xlabel("Training iterations")
    plt.show()

def plotCEGraph(ce_array, fold):
    plt.title("Cross-validation fold: {}\nTraining logistic regression model".format(fold))
    plt.plot(np.arange(len(ce_array)), ce_array)
    plt.ylabel("Cross-entropy")
    plt.xlabel("Training iteration")
    plt.show()


def crossValidation(train_set, test_set, folds, K, d):

    aa_per_fold = np.zeros(folds)
    weights_per_fold = np.zeros((folds, K, d))

    random.shuffle(train_set)
    random.shuffle(test_set)

    train_set_split = np.split(train_set, folds, axis=0)

    for k in range(folds):
        print("Cross-validation fold {} of {}:".format(k+1, folds))
        k_test_set = np.array(train_set_split[k])
        k_train_set = np.concatenate(np.delete(train_set_split, k, axis=0))

        k_train_x = np.array(k_train_set[:, 0:d]) / pixel_range
        k_train_y = np.array(k_train_set[:, d])
        k_test_x = np.array(k_test_set[:, 0:d]) / pixel_range
        k_test_y = np.array(k_test_set[:, d])

        k_train_y_binary = np.zeros((len(k_train_set), K))
        k_train_y_binary[np.arange(len(k_train_set)), k_train_y] = 1


        weights, cross_entropy, correct_classification = logistic_regression(k_train_x, k_train_y_binary,
                                                                             K, d,
                                                                             100, 0.0001)
        print("\n\tTraining finished")

        plotCCGraph(correct_classification, k)
        plotCEGraph(cross_entropy, k)

        k_predicted_classes = np.argmax(1 / (1 + np.exp(-(weights.dot(k_test_x.transpose())))), axis=0)

        aa_per_fold[k] = averageAccuracy(k_test_y, k_predicted_classes, K)
        weights_per_fold[k] = weights

    test_x = np.array(test_set[:, 0:d]) / pixel_range
    test_y = np.array(test_set[:, d])

    best_weights = weights_per_fold[np.argmax(aa_per_fold)]
    predicted_classes_best_model = np.argmax(1 / (1 + np.exp(-(best_weights.dot(test_x.transpose())))), axis=0)
    aa_best_model = averageAccuracy(test_y, predicted_classes_best_model, K)

    plotAABarPlot(aa_per_fold)

    return best_weights, aa_best_model


decision_boundaries, testing_accuracy = crossValidation(data_train, data_test, 5, number_of_classes, number_of_dimensions)

