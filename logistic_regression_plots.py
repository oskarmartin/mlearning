import numpy as np
import matplotlib.pyplot as plt


# Create bar plot of model average accuracy per cross-validation fold
def plotAABarPlot(aa_array, plotTitle, it, ss):
    fig = plt.figure()
    folds = len(aa_array)
    plt.title(plotTitle)
    plt.bar(np.arange(folds), aa_array * 100, 0.35)
    for f in range(folds):
        plt.annotate("{}%".format(aa_array[f] * 100), (f - 0.2, aa_array[f] * 100 + 1))
    plt.ylabel("Average model accuracy (%)")
    plt.xlabel("CV fold")
    plt.xticks(np.arange(folds), np.arange(1, 6))
    fig.savefig('plots/testing_performance_it{}_ss{}.png'.format(it, ss))
    plt.show()

# Create graph of model accuracy over training iterations
def plotCCGraph(cc_array, fold, it, ss):
    fig = plt.figure()
    plt.title("Training logistic regression model\nCross-validation fold: {}\nNumber of iterations: {}\nStep size: {}".format(fold+1, it, ss))
    plt.plot(np.arange(len(cc_array)), cc_array, color='orange')
    plt.ylabel("Correct classifications (%)")
    plt.xlabel("Training iteration")
    fig.savefig('plots/cc_training_performance_fold{}_it{}_ss{}.png'.format(fold+1, it, ss))
    plt.show()

# Create graph of model cross-entropy over training iterations
def plotCEGraph(ce_array, fold, it, ss):
    fig = plt.figure()
    plt.title("Training logistic regression model\nCross-validation fold: {}\nNumber of iterations: {}\nStep size: {}".format(fold+1, it, ss))
    plt.plot(np.arange(len(ce_array)), ce_array)
    plt.ylabel("Cross-entropy")
    plt.xlabel("Training iteration")
    fig.savefig('plots/ce_training_performance_fold{}_it{}_ss{}.png'.format(fold+1, it, ss))
    plt.show()

# Create box plot for the performance of all model variations
def plotAABoxPlot(aa_per_fold_per_it_value, aa_per_fold_per_ss_value, number_of_param_values):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Model performance")
    data_list = []
    x_labels = []

    # Collect average accuracies for cross-validation folds sharing the same number of iterations
    # Each collection is considered a model
    for i in range(number_of_param_values):
        model_i = aa_per_fold_per_it_value[:, i]
        data_list.append(model_i)
        x_labels.append("Model {}".format(i+1))

    # Collect average accuracies for cross-validation folds sharing the same step size
    # Each collection is considered a model
    for i in range(number_of_param_values):
        model_i = aa_per_fold_per_ss_value[:, i]
        data_list.append(model_i)
        x_labels.append("Model {}".format(number_of_param_values+i+1))

    bp = ax.boxplot(data_list)
    ax.set_xticklabels(x_labels)

    # Plot graphics settings
    for flier in bp['fliers']:
        flier.set(marker = 'o', color = 'pink', alpha = 0.5)
    for box in bp['boxes']:
        # change outline color
        box.set(color = 'pink', linewidth = 2)
        # change fill color
        box.set(color = 'tab:blue')

    fig.savefig('plots/aa_box_plot.png')
    plt.show()

