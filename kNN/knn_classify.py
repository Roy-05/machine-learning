from sys import argv
import numpy as np
import math

script, pendigits_training, pendigits_test, k = argv

training_file = open(f"{pendigits_training}.txt", 'r')
test_file = open(f"{pendigits_test}.txt", 'r')

training_dataset = []
for line in training_file:
    training_dataset.append([int(datapoint) for datapoint in line.split()])


test_dataset = []
for line in test_file:
    test_dataset.append([int(datapoint) for datapoint in line.split()])

def getMeanAndStd(training_dataset):
    meanAndStd = []
    for i in range(len(training_dataset[0])-1):
        column = [row[i] for row in training_dataset]
        mean = sum(column)/len(column)
        sigma = 0
        for datapoint in column:
            sigma += abs((datapoint - mean)**2)
        
        std = math.sqrt(sigma/len(column))
        meanAndStd.append({"mean": mean, "std": std})

    return meanAndStd

def normalizeData(training_dataset):
    meanAndStd = getMeanAndStd(training_dataset)
    for i in range(len(training_dataset)):
        for j in range(len(training_dataset[i])-1):
            mean = meanAndStd[j]["mean"]
            std = meanAndStd[j]["std"]
            training_dataset[i][j] = (training_dataset[i][j] - mean)/std

def euclidianDistance(row1, row2):
    dist = 0.0
    for i in range(len(row1)-1):
        dist += (row1[i] - row2[i])**2
    
    return math.sqrt(dist)

def getNeighbors(train, test_row):
    distances = list()
    for train_row in train:
        dist = euclidianDistance(train_row, test_row)
        distances.append([train_row, dist])

    distances.sort(key=lambda row:row[1])
    neighbors = list()

    for i in range(int(k)):
        neighbors.append(distances[i][0])

    return neighbors

def predictClass(train, test_row):
    neighbors = getNeighbors(training_dataset, test_row)
    output_vals = [row[-1] for row in neighbors]
    prediction = max(set(output_vals), key=output_vals.count)

    return prediction

for row in test_dataset:
    print(f"Actual class {row[-1]}, Predicted Class {predictClass(training_dataset, row)}")

