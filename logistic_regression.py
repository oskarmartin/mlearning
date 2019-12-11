import numpy as np
import logistic_regression_plots as lg_p
import ml_tools as ml


# Logistic regression model:
# Train model on input data and output decision boundary weights and model performance
def performLogisticRegression(X, r, K, d, iterations, eta):

    train_set_size = len(X)
    W = np.random.uniform(-0.01, 0.01, (K, d))
    y = np.zeros((train_set_size, K))
    cross_e = np.zeros(iterations)
    acc = np.zeros(iterations)

    for it in range(iterations):
        print("\r\t\tRunning training iteration {} of {}".format(it+1, iterations)),
        grad_w = np.zeros((K, d))

        for t in range(train_set_size):
            # Estimate data label using current weights
            o = (W * X[t]).sum(axis=1)
            y[t] = np.exp(o) / np.sum(np.exp(o))

            # Update the gradient weights
            for i in range(K):
                grad_w[i] += (r[t][i] - y[t][i]) * X[t]

        # Perform a gradient descent step of size eta
        W += eta * grad_w

        # Training accuracy
        pred_labels = np.argmax(y, axis=1)
        true_labels = np.argmax(r, axis=1)
        acc[it] = ml.averageAccuracy(true_labels, pred_labels, K)
        # Cross entropy
        prob_correct_class = np.choose(true_labels, y.T)
        cross_e[it] = -sum(np.log(prob_correct_class)/float(train_set_size))

    return W, cross_e, acc


# Call for logistic regression training and testing
def train_and_test_lg(train_x, train_y_binary, K, d, it, ss, test_x, test_y):

    # Send training data through the logistic regression model
    weights, cross_entropy, correct_classification = performLogisticRegression(train_x, train_y_binary, K, d, it, ss)
    print("\n")

    # Evaluate the resulting weights on testing data
    predicted_classes = np.argmax(1 / (1 + np.exp(-(weights.dot(test_x.transpose())))), axis=0)
    aa_per_fold_per_ss_value = ml.averageAccuracy(test_y, predicted_classes, K)

    return weights, correct_classification, cross_entropy, aa_per_fold_per_ss_value


# Cross validation (CV):
# Perform "folds"-folds cross-validation on input training set
# with "number_of_param_values" number of training iterations and step size respectively,
# yielding "number_of_param_values" * 2 models, all tested using CV.
# Evaluate optimal number of training iterations and step size and
# use them for final testing with input test set
def log_reg_cross_validation(train_set, test_set, folds, K, d, iteration_values, step_size_values):

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
        k_train_x, k_train_y, k_test_x, k_test_y, k_train_y_binary = ml.splitIntoFeaturesAndLabels(k_test_set, k_train_set, K, d)

        # Varying parameter-values loop
        for i in range(number_of_param_values):

            print("\tParameter change {} of {}".format(i+1, number_of_param_values))

            # Train model on fold k training data with varying number of training iterations
            # and test the results on fold k test data
            weights_i, correct_classification_i, cross_entropy_i, aa_per_fold_per_it_value[k][i] = train_and_test_lg(k_train_x, k_train_y_binary,
                                                                                                                     K, d, iteration_values[i],
                                                                                                                     default_step_size, k_test_x, k_test_y)
            # Train model on fold k training data with varying number of step size
            # and test the results on fold k test data
            weights_s, correct_classification_s, cross_entropy_s, aa_per_fold_per_ss_value[k][i] = train_and_test_lg(k_train_x, k_train_y_binary,
                                                                                                                     K, d, default_iteration_size,
                                                                                                                     step_size_values[i], k_test_x, k_test_y)
            # Plot training accuracy for the two models defined above
            lg_p.plotCCGraph(correct_classification_i, k, iteration_values[i], default_step_size)
            lg_p.plotCCGraph(correct_classification_s, k, default_iteration_size, step_size_values[i])

            # Plot training cross-entropy for the two models defined above
            lg_p.plotCEGraph(cross_entropy_i, k, iteration_values[i], default_step_size)
            lg_p.plotCEGraph(cross_entropy_s, k, default_iteration_size, step_size_values[i])

    # Create bar plot for every model showing validation accuracy for each CV-fold
    for i in range(number_of_param_values):

        lg_p.plotAABarPlot(aa_per_fold_per_it_value[:, i], "\nModel {}\nTraining iterations = {}, step size = {}\n".format(i + 1, iteration_values[i], default_step_size), iteration_values[i], default_step_size)
        lg_p.plotAABarPlot(aa_per_fold_per_ss_value[:, i], "\nModel {}\nTraining iterations = {}, step size = {}\n".format(number_of_param_values + i + 1, default_iteration_size, step_size_values[i]), default_iteration_size, step_size_values[i])

    # Compute for every model the average validation accuracy mean
    mean_ave_acc_i = np.mean(aa_per_fold_per_it_value, axis = 0)
    mean_ave_acc_s = np.mean(aa_per_fold_per_ss_value, axis = 0)

    # Compute for evert model the average validation accuracy variance
    var_ave_acc_i = np.var(aa_per_fold_per_it_value, axis = 0)
    var_ave_acc_s = np.var(aa_per_fold_per_ss_value, axis = 0)

    # Find optimal number of training iterations based on best model performance
    optimal_iteration_size_mean = iteration_values[np.argmax(mean_ave_acc_i)]
    # Find optimal step size based on best model performance
    optimal_step_size_mean = step_size_values[np.argmax(mean_ave_acc_s)]

    # Find optimal number of training iterations based on lowest model variance
    optimal_iteration_size_var = iteration_values[np.argmin(var_ave_acc_i)]
    # Find optimal step size based on lowest model variance
    optimal_step_size_var = step_size_values[np.argmin(var_ave_acc_s)]

    # Create box plot comparing model performances
    lg_p.plotAABoxPlot(aa_per_fold_per_it_value, aa_per_fold_per_ss_value, number_of_param_values)

    # Create bar plot comparing model performances
    lg_p.plotCollectedModelsAABarPlot(mean_ave_acc_i, mean_ave_acc_s, iteration_values, step_size_values, "Mean average validation accuracy over CV-folds", 1, "it")
    lg_p.plotCollectedModelsAABarPlot(var_ave_acc_i, var_ave_acc_s, iteration_values, step_size_values, "Average validation accuracy variance over CV-folds", 3, "ss")

    # Final testing:
    # Separate full training and test data into attributes and class labels
    train_x, train_y, test_x, test_y, train_y_binary = ml.splitIntoFeaturesAndLabels(test_set, train_set, K, d)

    print("Final training")

    # Train logistic regression model on all training data, and test with all test data.
    # Number of training iteration and step size is derived from model of best performance.
    _, _, _, final_test_aa_opt_mean = train_and_test_lg(train_x, train_y_binary,
                                                        K, d, optimal_iteration_size_mean,
                                                        optimal_step_size_mean, test_x, test_y)
    # Train logistic regression model on all training data, and test with all test data.
    # Number of training iteration and step size is derived from model of lowest variance.
    _, _, _, final_test_aa_opt_var = train_and_test_lg(train_x, train_y_binary,
                                                       K, d, optimal_iteration_size_var,
                                                       optimal_step_size_var, test_x, test_y)

    testing_aa_per_model = np.zeros(number_of_param_values * 2)
    for i in range(number_of_param_values):
        _, training_aa, _, testing_aa_per_model[i] = train_and_test_lg(train_x, train_y_binary,
                                                             K, d, iteration_values[i],
                                                             default_step_size, test_x, test_y)
        if i == 3:
            print("Training accuracy (Model 4): {}".format(training_aa))

        _, _, _, testing_aa_per_model[number_of_param_values + i] = train_and_test_lg(train_x, train_y_binary,
                                                                                      K, d, default_iteration_size,
                                                                                      step_size_values[i], test_x, test_y)

    print("Average testing accuracy: Model 1-8 = {}\n".format(testing_aa_per_model))

    print("Mean average validation accuracy for model 1-4: {}".format(mean_ave_acc_i))
    print("Mean average validation accuracy for model 5-8: {}".format(mean_ave_acc_s))
    print("Average validation accuracy variance for model 1-4: {}".format(var_ave_acc_i))
    print("Average validation accuracy variance for model 5-8: {}".format(var_ave_acc_s))

    print("Average testing accuracy (parameters with highest model performance): {}".format(final_test_aa_opt_mean))
    print("Average testing accuracy (parameters with lowest model variance): {}".format(final_test_aa_opt_var))
