import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import trange
from IPython.display import clear_output
import ml_tools as ml
import os


number_of_classes = 5
number_of_dimensions = 28 * 28
pixel_range = 255.0
np.random.seed(42)

data_train = np.load("data/fashion_train.npy")
data_test = np.load("data/fashion_test.npy")

class Layer:
    """
        Building block. Each layer is performing 2 things:
            - Process input to get output
            - Propagate gradients through itself
        
        Some layers have learnable parameters (if any present) which
        they update during layer.backward
    """
    def __init__(self):
        pass
    
    def forward(self, input):
        return input
    
    def backward(self, input, grad_output):
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)

        return np.dot(grad_output, d_layer_d_input)

class ReLu(Layer):
    def __init__(self):
        pass
    def forward(self, input):
        relu_forward = np.maximum(0, input)
        return relu_forward
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2/(input_units+output_units)), size = (input_units, output_units))
        self.biases = np.zeros(output_units)

    def forward(self, input):
        return np.dot(input, self.weights) + self.biases
    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

def softmax_crossentropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]
    xentropy = -logits_for_answers + np.log(np.sum(np.exp(logits), axis = -1))
    return xentropy

def grad_softmax_crossentropy_with_logits(logits, reference_answers):

    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_for_answers + softmax) / logits.shape[0]


def load_dataset(flatten=False, extended_data=False):

    #data_train = ml.extendData(data_train, num_of_dimensions)
    #data_test = ml.extendData(data_test, num_of_dimensions)

    random.shuffle(data_train)
    num_of_dimensions = 28 * 28
    
    if extended_data:

        train_x = np.array(ml.extendData(data_train, num_of_dimensions)[:, 0:num_of_dimensions])
        train_y = np.array(ml.extendData(data_train, num_of_dimensions)[:, num_of_dimensions])
    
        test_x = np.array(ml.extendData(data_test, num_of_dimensions)[:, 0:num_of_dimensions])
        test_y = np.array(ml.extendData(data_test, num_of_dimensions)[:, num_of_dimensions])
    else:

        train_x = np.array(data_train[:, 0:num_of_dimensions])
        train_y = np.array(data_train[:, num_of_dimensions])
    
        test_x = np.array(data_test[:, 0:num_of_dimensions])
        test_y = np.array(data_test[:, num_of_dimensions])

    #normalize
    test_x = test_x.astype(float) / pixel_range
    train_x = train_x.astype(float) / pixel_range

    train_x, validation_x = train_x[:-7500], train_x[-2500:]
    train_y, validation_y = train_y[:-7500], train_y[-2500:]

    if flatten:
        train_x = train_x.reshape([train_x.shape[0], -1])
        validation_x = validation_x.reshape([validation_x.shape[0], -1])
        test_x = test_x.reshape([test_x.shape[0], -1])

    return train_x, train_y, validation_x, validation_y, test_x, test_y


train_x, train_y, validation_x, validation_y, test_x, test_y = load_dataset(flatten=True, extended_data=True)

#NETWORK

network = []
network.append(Dense(train_x.shape[1], 100))
network.append(ReLu())
network.append(Dense(100, 200))
network.append(ReLu())
network.append(Dense(200, 10))


def forward(network, X):
    #Compute activations of all network layers by applying them sequentially
    #Return a list of activations for each layer
    activations = []
    input = X

    #Loop through each layer
    for layer in network:
        activations.append(layer.forward(input))
        
        #Updating input to last layer output
        input = activations[-1]

    assert len(activations) == len(network)
    return activations

def predict(network, X):
    #Compute network predictions. Returning indices of largest Logit probability

    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)

def train(network, X, y):
    #Train network on the given batch of X and y
    #Firstly need to run forward to get all layer activations
    #Then can run layer.backward going from last to first layer
    #After backward is called for all layers, all Dense layers have already made one gradient step

    #Get layer activations
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations # layer_inputs[i] is input for network[i]
    logits = layer_activations[-1]

    #Compute the loss and initial gradient
    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)


    #Propagate gradients through the network
    #Reverse propagation as this is  backprop
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]

        loss_grad = layer.backward(layer_inputs[layer_index], loss_grad) 

    return np.mean(loss)



#TRAINING LOOP

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx+batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
    

train_log = []
validation_log = []

for iteration in range(10):
    mean_accuracy = []
    for epoch in range(25):
        
        for x_batch, y_batch in iterate_minibatches(train_x, train_y, batchsize=200, shuffle=True):
            train(network, x_batch, y_batch)
        
        train_log.append(np.mean(predict(network, train_x) == train_y)) # 75 %
        
        validation_log.append(np.mean(predict(network, validation_x) == validation_y)) # 25 %

        clear_output()
        mean_accuracy.append(validation_log[-1])
        print("Train accuracy: ", train_log[-1])
        print("Validation accuracy: ", validation_log[-1])

        """
        print("Epoch ", epoch)
        print("Train accuracy: ", train_log[-1])
        print("Validation accuracy: ", validation_log[-1])
        
        if epoch == 1:
            plt.plot(train_log, label="Training accuracy")
            plt.plot(validation_log, label="Validation accuracy")
            plt.legend(loc="best")
        #plt.grid()
        #plt.show()
        
        if epoch == 24:
            if not os.path.exists('./perceptron/' + str(iteration)):
                os.makedirs('./perceptron/' + str(iteration))
            plt.savefig('./perceptron/' + str(iteration) + '/normal.png')
        """

    print(mean_accuracy)
    print("Average Validation Accuracy after 10 iterations: " + str(np.mean(mean_accuracy)))


    
"""

average_perf = [0.8594720000000001 * 100, 0.905632 * 100]
labels = ('Normal data', 'Extended data')
y_pos = np.arange(len(labels))
plt.bar(y_pos, average_perf, width=0.3)
plt.xticks([])
plt.ylabel("Percentage %")

plt.title("Normal & Extended data comparison")


plt.show()
"""