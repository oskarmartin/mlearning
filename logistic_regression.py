import numpy as np
import logistic_regression_plots as lgp
import data_processing as dp


# Logistic regression model:
# Train model on input data and output decision boundary weights and model performance
def logistic_regression(x, r, K, d, iterations, eta):

    train_set_size = len(x)
    w = np.random.uniform(-0.01, 0.01, (K, d))
    y = np.zeros((train_set_size, K))
    cross_e = np.zeros(iterations)
    acc = np.zeros(iterations)

    for it in range(iterations):
        print("\r\t\t\tRunning iteration {} of {}".format(it+1, iterations)),
        grad_w = np.zeros((K, d))

        for t in range(train_set_size):
            o = (w * x[t]).sum(axis=1)
            y[t] = np.exp(o) / np.sum(np.exp(o))

            if np.argmax(y[t]) == np.argmax(r[t]):
                acc[it] += 1.0/float(train_set_size)

            for i in range(K):
                grad_w[i] += (r[t][i] - y[t][i]) * x[t]

        w += eta * grad_w

        # Training accuracy
        correct_class_indices = np.argmax(r, axis=1)
        # Cross entropy
        prob_correct_class = np.choose(correct_class_indices, y.T)
        cross_e[it] = -sum(np.log(prob_correct_class)/float(train_set_size))

    return w, cross_e, acc


# Average accuracy:
# Compute average accuracy based on set of true and predicted classes
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

# Train and test logistic regression model
def train_and_test(train_x, train_y_binary, K, d, it, ss, test_x, test_y):

    # Send training data through the logistic regression model
    print("\t\tTraining initialized")
    weights, cross_entropy, correct_classification = logistic_regression(train_x, train_y_binary, K, d, it, ss)
    print("\n\t\tTraining finished")

    # Evaluate the resulting weights on testing data
    predicted_classes = np.argmax(1 / (1 + np.exp(-(weights.dot(test_x.transpose())))), axis=0)
    aa_per_fold_per_ss_value = averageAccuracy(test_y, predicted_classes, K)

    return weights, correct_classification, cross_entropy, aa_per_fold_per_ss_value


# Cross validation (CV):
# Perform "folds"-folds cross-validation on input training set
# with "number_of_param_values" number of training iterations and step size respectively,
# yielding "number_of_param_values" * 2 models, all tested using CV.
# Evaluate optimal number of training iterations and step size and
# use them for final testing with input test set
def crossValidation(train_set, test_set, folds, K, d, iteration_values, step_size_values, element_range):

    # Number of different parameter values to model after
    number_of_param_values = len(iteration_values)
    # Default parameter values (when one is varied the other is fixed)
    default_iteration_size = iteration_values[number_of_param_values / 2]
    default_step_size = step_size_values[number_of_param_values / 2]

    aa_per_fold_per_it_value = np.zeros((folds, len(iteration_values)))
    aa_per_fold_per_ss_value = np.zeros((folds, len(step_size_values)))

    # Split training data into "folds" set
    train_set_split = np.split(train_set, folds, axis=0)

    # CV loop
    for k in range(folds):
        print("*** Cross-validation fold {} of {} ***".format(k+1, folds))

        # Pick 1 split training data subset for testing and 4 subsets for training
        k_test_set = np.array(train_set_split[k])
        k_train_set = np.concatenate(np.delete(train_set_split, k, axis=0))

        # Separate attributes from class label in the fold k training and test set
        k_train_x, k_train_y, k_test_x, k_test_y, k_train_y_binary = dp.splitIntoAttAndClass(k_test_set, k_train_set, K, d, element_range)

        # Varying parameter-values loop
        for i in range(number_of_param_values):

            print("\tParameter change {} of {}".format(i+1, number_of_param_values))

            # Train model on fold k training data with varying number of training iterations
            # and test the results on fold k test data
            weights_i, correct_classification_i, cross_entropy_i, aa_per_fold_per_it_value[k][i] = train_and_test(k_train_x, k_train_y_binary,
                                                                                                                  K, d, iteration_values[i],
                                                                                                                  default_step_size, k_test_x, k_test_y)

            # Train model on fold k training data with varying number of step size
            # and test the results on fold k test data
            weights_s, correct_classification_s, cross_entropy_s, aa_per_fold_per_ss_value[k][i] = train_and_test(k_train_x, k_train_y_binary,
                                                                                                                  K, d, default_iteration_size,
                                                                                                                  step_size_values[i], k_test_x, k_test_y)
            # Plot training accuracy for the two models defined above
            lgp.plotCCGraph(correct_classification_i, k, iteration_values[i], default_step_size)
            lgp.plotCCGraph(correct_classification_s, k, default_iteration_size, step_size_values[i])

            # Plot training cross-entropy for the two models defined above
            lgp.plotCEGraph(cross_entropy_i, k, iteration_values[i], default_step_size)
            lgp.plotCEGraph(cross_entropy_s, k, default_iteration_size, step_size_values[i])

    # Create bar plot for every model showing testing accuracy for each CV-fold
    for i in range(number_of_param_values):

        lgp.plotAABarPlot(aa_per_fold_per_it_value[:, i], "Testing logistic regression model\nTraining iterations = {}, step size = {}".format(iteration_values[i], default_step_size), iteration_values[i], default_step_size)
        lgp.plotAABarPlot(aa_per_fold_per_ss_value[:, i], "Testing logistic regression model\nTraining iterations = {}, step size = {}".format(default_iteration_size, step_size_values[i]), default_iteration_size, step_size_values[i])

    # Compute for evert model the average accuracy mean
    ave_mean_acc_i = np.mean(aa_per_fold_per_it_value, axis = 0)
    ave_mean_acc_s = np.mean(aa_per_fold_per_ss_value, axis = 0)

    # Compute for evert model the average accuracy variance
    ave_var_acc_i = np.var(aa_per_fold_per_it_value, axis = 0)
    ave_var_acc_s = np.var(aa_per_fold_per_ss_value, axis = 0)

    # Find optimal number of training iterations based on best model performance
    optimal_iteration_size_mean = iteration_values[np.argmax(ave_mean_acc_i)]
    # Find optimal number of training iterations based on lowest model variance
    optimal_step_size_mean = step_size_values[np.argmax(ave_mean_acc_s)]

    # Find optimal step size based on best model performance
    optimal_iteration_size_var = iteration_values[np.argmin(ave_var_acc_i)]
    # Find optimal step size based on lowest model variance
    optimal_step_size_var = step_size_values[np.argmin(ave_var_acc_s)]

    # Create box plot comparing model performances
    lgp.plotAABoxPlot(aa_per_fold_per_it_value, aa_per_fold_per_ss_value, number_of_param_values)

    # Final testing:
    # Separate full training and test data into attributes and class labels
    train_x, train_y, test_x, test_y, train_y_binary = dp.splitIntoAttAndClass(test_set, train_set, K, d, element_range)

    print("Final training")

    # Train logistic regression model on all training data, and test with all test data.
    # Number of training iteration and step size is derived from model of best performance.
    weights_opt_mean, _, _, final_test_aa_opt_mean = train_and_test(train_x, train_y_binary,
                                                                    K, d, optimal_iteration_size_mean,
                                                                    optimal_step_size_mean, test_x, test_y)
    # Train logistic regression model on all training data, and test with all test data.
    # Number of training iteration and step size is derived from model of lowest variance.
    weights_opt_var, _, _, final_test_aa_opt_var = train_and_test(train_x, train_y_binary,
                                                                  K, d, optimal_iteration_size_var,
                                                                  optimal_step_size_var, test_x, test_y)

    print("Average accuracy - final testing: {}".format(final_test_aa_opt_mean))
    print("Average accuracy - final testing: {}".format(final_test_aa_opt_var))


