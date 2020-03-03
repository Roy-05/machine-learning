import os
from math import exp
from random import random


training_file = os.getcwd() + r'/ANN/data/optdigits-3.tra'
test_file     = os.getcwd() + r'/ANN/data/optdigits-3.tes'

neural_net = []
training_dataset, test_dataset = [],[]
training_set, validation_set = [],[]
training_mse, validation_mse = [],[]
learning_rate = 0.002
epochs = 2000

# Create dataset from file
def create_dataset(filename, dataset):
    f = open(filename, 'r')
    for line in f:
        line = line.rstrip('\n').split(',')
        # Normalize Data
        dataset.append([(int(line[i])/16.0) if i != len(line) - 1 else int(line[i]) for i in range(len(line))])
    f.close()


# Create initial nn with 1 hidden layer and 1 output layer
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
    # Initialize with bias
    sigma = weights[-1]
    for i in range(len(weights)-1):
        sigma += weights[i]*inputs[i]
    
    return sigma


#Forward propagate input to the network output
def forward_propagation(row):
    inputs = row
    for layer in neural_net:
        new_inputs = []
        for neuron in layer:
            y = feedForward(neuron['w'], inputs)
            neuron['output'] = activate(y)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative
def sigmoid_prime(y):
    return y * (1.0 - y)


# Back propagate errors
def back_propagation(row, expected):
    for i in reversed(range(len(neural_net))):
        layer = neural_net[i]
        inputs = row[:-1] if i==0 else [neuron['output'] for neuron in neural_net[i-1]]
 
        # Case: Output layer
        if i == (len(neural_net) - 1):
            for neuron,j in zip(layer,range(len(layer))):   
                error = expected[j] - neuron["output"]
                neuron['delta'] = error * sigmoid_prime(neuron['output'])

                # Update weights
                for k in range(len(inputs)):
                    neuron['w'][k] += learning_rate * neuron['delta'] * inputs[k]
                neuron['w'][-1] += learning_rate * neuron['delta'] # Update Bias
        else:
            for neuron,j in zip(layer,range(len(layer))):   
                error = 0.0
                for next_layer_neuron in neural_net[i+1]:
                    error += next_layer_neuron['w'][j] * next_layer_neuron['delta']

                neuron['delta'] = error * sigmoid_prime(neuron['output'])
                
                # Update weights
                for k in range(len(inputs)):
                    neuron['w'][k] += learning_rate * neuron['delta'] * inputs[k]
                neuron['w'][-1] += learning_rate * neuron['delta'] # Update Bias
        

# Train nn across epochs
def train_nn(tra_dataset, tes_dataset, tra_arr, tes_arr):
    for epoch in range(epochs):
        tra_mse = 0.0
        tes_mse = 0.0
        for row in tra_dataset:
            forward_propagation(row)

            expected_vector = [0.1, 0.1, 0.1, 0.1]
            expected_vector[row[-1]] = 0.9

            back_propagation(row, expected_vector)

        if (epoch%10 == 0):
            tra_mse = mean_square_error(tra_dataset, tra_arr)
            tes_mse = mean_square_error(tes_dataset, tes_arr)
    
            print(f"Mean Squared Error on Training Set: {tra_mse}")
            print(f"Mean Squared Error on Training Set: {tes_mse}")
            print(f"Epoch                             : {epoch}\n")

# Get mse for the current epoch
def mean_square_error(dataset, arr):
    mse = 0.0
    for row in dataset:
        expected_vector = [0.1, 0.1, 0.1, 0.1]
        expected_vector[row[-1]] = 0.9
        for i in range(4):
            mse += (expected_vector[i] - neural_net[1][i]['output'])**2

    mse = round(mse/len(dataset), 10)
    arr.append(mse)
    return mse

# Make predictions after adjusting weights
def predict(row):
    outputs = forward_propagation(row)
    return outputs.index(max(outputs))


# Driver code
def main():

    # Create our training dataset from the file by invoking this function
    create_dataset(training_file, training_dataset)

    # Create our test dataset from the file by invoking this function
    create_dataset(test_file, test_dataset)
    
    # Initialize our neural net by invoking this function
    initialize_neural_network(64, 8, 4)
    training_rows = round(len(training_dataset)*0.8)
    training_set = training_dataset[:training_rows]
    validation_set = training_dataset[training_rows:]

    train_nn(training_set, validation_set, training_mse, validation_mse)

    accuracy = 0.0
    for row in test_dataset:
        actual = row[-1]
        predicted = predict(row)
        accuracy += 1.0 if actual == predicted else 0.0
        # print(f"Expected:{actual:2d} Actual:{predicted:2d}")

    print(f"\nAccuracy: {accuracy/len(test_dataset):2.5f}\n")
    
main()
