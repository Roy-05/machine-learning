import os
from random import random

neural_net = list()
dataset = list()
filename = os.getcwd() + r'/ANN/data/optdigits-3.tra'


def create_dataset(filename):
    f = open(filename, 'r')
    for line in f:
        line = line.rstrip('\n')
        dataset.append([line])
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

# Create our dataset from the file by invoking this function
create_dataset(filename)

# Initialize our neural net by invoking this function
initialize_neural_network(64, 8, 4)
