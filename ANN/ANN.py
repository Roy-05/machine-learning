import os
from math import exp
from random import random


filename = os.getcwd() + r'/ANN/data/optdigits-3.tra'

neural_net, dataset, training_set, test_set = [],[],[],[]
learning_rate = 0.01
epochs = 100


def create_dataset(filename):
    f = open(filename, 'r')
    for line in f:
        line = line.rstrip('\n').split(',')
        
        dataset.append([(int(line[i])/16.0) if i != len(line) - 1 else int(line[i]) for i in range(len(line))])
    f.close()


def initialize_neural_network(n_inputs, n_hidden, n_outputs):
    """
    n_inputs: number of neurons in input layer\n
    n_hidden: number of neurons in hidden layer\n
    n_outputs: number of neurons in output layer\n
    """

    # +1 to add the bias
    hidden_layer = [{'w': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    neural_net.append(hidden_layer)

    output_layer = [{'w':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    neural_net.append(output_layer)


# Activation function is a sigmoid function
# sigmoid = 1/(1 + e^-y)
def activate(y):
    return 1.0/(1.0 + exp(-y))


def feedForward(weights, inputs):
    # Initialize with bias which is the last element in the weights list
    sigma = weights[-1]
    for i in range(len(weights)-1):
        sigma += weights[i]*inputs[i]
    
    return sigma


#Forward propagate input to the network output
def forward_propagation(row):
    for layer in neural_net:
        new_inputs = list()
        inputs = row
        for neuron in layer:
            y = feedForward(neuron['w'], inputs)
            neuron['output'] = activate(y)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of a neuron output
def sigmoid_prime(y):
    return y * (1.0 - y)


def back_propagation(expected):
    for i in range(len(neural_net)-1, -1, -1):
        layer = neural_net[i]
        errors = []
        q = len(layer)

        # Case: Output layer
        if i == (len(neural_net) - 1):
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron["output"])
        else:
            for j in range(len(layer)):
                error = 0.0
                for neuron in neural_net[i+1]:
                    error += neuron['w'][j] * neuron['delta']
                errors.append(error)
        

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * activate(neuron['output'])


def update_weights(row):
    for i in range(len(neural_net)):
        inputs = row[:-1]
        if(i != 0):
            inputs = [neuron['output'] for neuron in neural_net[i-1]]
        for neuron in neural_net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta'] # Update Bias


def train_nn():
    for epoch in range(epochs):
        for row in dataset:
            # outputs = forward_propagation(row)

            expected_vector = [0.1, 0.1, 0.1, 0.1]
            expected_vector[row[-1]] = 0.9

            back_propagation(expected_vector)
            update_weights(row)


def predict(row):
    outputs = forward_propagation(row)
    return outputs.index(max(outputs))

# Create our dataset from the file by invoking this function
create_dataset(filename)

# Initialize our neural net by invoking this function
initialize_neural_network(64, 2, 4)

for row in dataset:
    prediction = predict(row)
    print(f"Expected:{row[-1]:3d} Actual:{prediction:3d} ")