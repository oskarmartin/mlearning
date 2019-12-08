import numpy as np
import matplotlib.pyplot as plt

scatter_shapes = ['o', 'v', 's', '*', 'x']
colors = ['gray', 'tab:purple', 'tab:blue', 'orange', 'pink']


# Create bar plot of model average accuracy per cross-validation fold
def plotAABarPlot(aa_array, plotTitle, it, ss):
    fig = plt.figure(figsize=(8, 6))
    folds = len(aa_array)
    plt.title(plotTitle)
    plt.bar(np.arange(folds), aa_array * 100, 0.35)
    for f in range(folds):
        plt.annotate("{}%".format(aa_array[f] * 100), (f - 0.2, aa_array[f] * 100 + 1))
    plt.ylabel("Average validation accuracy(%)")
    plt.xlabel("Cross-validation fold")
    plt.xticks(np.arange(folds), np.arange(1, 6))
    fig.savefig('logisticRegPlots/testing_performance_it{}_ss{}.png'.format(it, ss))
    plt.show()


# Create graph of model accuracy over training iterations
def plotCCGraph(cc_array, fold, it, ss):
    fig = plt.figure()
    plt.title("Training logistic regression model\nCross-validation fold: {}\nNumber of iterations: {}\nStep size: {}".format(fold+1, it, ss))
    plt.plot(np.arange(len(cc_array)), cc_array, color='orange')
    plt.ylabel("Correct classifications (%)")
    plt.xlabel("Training iteration")
    fig.savefig('logisticRegPlots/cc_training_performance_fold{}_it{}_ss{}.png'.format(fold+1, it, ss))
    plt.show()


# Create graph of model cross-entropy over training iterations
def plotCEGraph(ce_array, fold, it, ss):
    fig = plt.figure()
    plt.title("Training logistic regression model\nCross-validation fold: {}\nNumber of iterations: {}\nStep size: {}".format(fold+1, it, ss))
    plt.plot(np.arange(len(ce_array)), ce_array)
    plt.ylabel("Cross-entropy (training accuracy)")
    plt.xlabel("Training iteration")
    fig.savefig('logisticRegPlots/ce_training_performance_fold{}_it{}_ss{}.png'.format(fold+1, it, ss))
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

    plt.ylabel("Average validation accuracy")
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

    fig.savefig('logisticRegPlots/aa_box_plot.png')
    plt.show()


# Create bar plot for the performance of all model variations
def plotCollectedModelsAABarPlot(bar_model_1_to_4, bar_model_5_to_8, it_range, ss_range, y_legend, col_int, plot_type):

    # Collect x-axis labels
    number_of_param_values = len(it_range)

    default_iteration_size = it_range[number_of_param_values / 2]
    default_step_size = ss_range[number_of_param_values / 2]

    it_labels = np.hstack((it_range, np.full(number_of_param_values, default_iteration_size))).astype(int)
    ss_labels = np.hstack((np.full(number_of_param_values, default_step_size), ss_range))

    # Set bar properties
    bar_width = 0.4
    bar = np.hstack((bar_model_1_to_4, bar_model_5_to_8))
    pos_bar = np.arange(number_of_param_values * 2)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.bar(pos_bar, bar, color=colors[col_int], width=bar_width, edgecolor='white')

    # Set bar and axis labels
    if plot_type == 'it':
        plt.ylim(np.min(bar) - np.std(bar), np.max(bar) + np.std(bar))

    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end + (end / 100), end - start))

    x_labels = []
    for i in range(len(pos_bar)):
        if plot_type == 'it':
            plt.text(x = pos_bar[i] - 0.21, y = bar[i] + 0.0005, s = "{:0.2f}%".format(bar[i] * 100), size = 9)
            ax.yaxis.set_ticklabels(["{:0.2f}%".format(start * 100), "{:0.2f}%".format((end + end / 100) * 100)])
        else:
            plt.text(x=pos_bar[i] - 0.31, y=bar[i] + 0.0000003, s="{:0.2e}".format(bar[i]), size=9)
            ax.yaxis.set_ticklabels(["0", "{:0.2e}".format(end + (end / 100))])
        x_labels.append(u'Model {}\nit = {}\n\u03B7 = {}'.format(i+1, it_labels[i], ss_labels[i]))

    plt.xticks([r for r in range(number_of_param_values * 2)], x_labels, size = 8)
    plt.ylabel(y_legend)
    plt.title("Model performances\n")

    fig.savefig('logisticRegPlots/collected_models_bar_plot_{}.png'.format(plot_type))

    plt.show()