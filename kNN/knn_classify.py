from sys import argv
import numpy as np
import math

script, pendigits_training, pendigits_test = argv

training_dataset = open(f"{pendigits_training}.txt", 'r')

matrix = []
for line in training_dataset:
    matrix.append(line.split())

np_arr = np.array(matrix).astype(np.float)
np_arr = np_arr.transpose()

def getStd(dimension):
    mean = np.mean(dimension)
    sigma = 0
    for datapoint in dimension:
        sigma += abs((datapoint - mean)**2)
    
    std = math.sqrt(sigma/len(dimension))

    return std

def euclidianDistance(vector1, vector2):
    dist = 0.0
    for i in range(len(vector1)-1):
        dist += (vector1[i] - vector2[i])**2
    
    return math.sqrt(dist)

def getNeighbors(train, test_row, k):
    distances = list()
    for train_row in train:
        dist = euclidianDistance(train_row, test_row)
        distances.append([train_row, dist])

    distances.sort(key=lambda row:row[1])
    neighbors = list()

    for i in range(k):
        neighbors.append(distances[i][0])

    return neighbors

for i in range(0, len(np_arr)-1):
    row = np_arr[i]
    std = getStd(row)
    mean = np.mean(row)
    for j in range(0, len(row)):
        row[j] = (row[j] - mean)/std

np_arr = np_arr.transpose()


neighbors = getNeighbors(np_arr, np_arr[0], 3)

for n in neighbors: 
    print(n)