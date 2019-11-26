import numpy as np
import matplotlib.pyplot as plt


colors = ['gray', 'tab:purple', 'tab:blue', 'orange', 'pink']

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
    plt.bar(pos_bar, bar, color=colors[col_int], width=bar_width, edgecolor='white')

    # Set plot labels
    x_labels = []
    for i in range(len(pos_bar)):
        if plot_type == 'it':
            plt.text(x = pos_bar[i] - 0.21, y = bar[i] + 0.0005, s = "{:0.2f}%".format(bar[i] * 100), size = 9)
        else:
            plt.text(x=pos_bar[i] - 0.31, y=bar[i] + 0.0000005, s="{:0.2e}".format(bar[i]), size=9)
        x_labels.append("Model {}\nit = {}\nss = {}".format(i+1, it_labels[i], ss_labels[i]))

    if plot_type == 'it':
        plt.ylim(np.min(bar) - np.std(bar), np.max(bar) + np.std(bar))

    plt.yticks([])
    plt.xticks([r for r in range(number_of_param_values * 2)], x_labels, size = 8)
    plt.ylabel(y_legend)
    plt.title("Model performances\n")

    fig.savefig('plots/collected_models_bar_plot_{}.png'.format(plot_type))

    plt.show()

plotCollectedModelsAABarPlot(np.array([0.9282,  0.9318,  0.9344, 0.93608]), np.array([0.9234, 0.92704, 0.935, 0.93028]), np.array([250, 300, 350, 400]), np.array([0.000015, 0.00001, 0.0000015, 0.000001]), "Mean average validation accuracy", 1, "it")
plotCollectedModelsAABarPlot(np.array([2.59520e-05, 2.31200e-05, 2.70720e-05, 2.39776e-05]), np.array([2.65760e-05, 1.37504e-05, 2.15680e-05, 3.24416e-05]), np.array([250, 300, 350, 400]), np.array([0.000015, 0.00001, 0.0000015, 0.000001]), "Average validation accuracy variance", 3, "ss")


