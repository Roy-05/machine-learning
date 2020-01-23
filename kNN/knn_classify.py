from sys import argv
import numpy as np
import math

script, pendigits_training, pendigits_test = argv

training_dataset = open(f"{pendigits_training}.txt", 'r')

matrix = []
for line in training_dataset:
    matrix.append(line.split())

np_arr = np.array(matrix).astype(np.float)
np_arr.transpose()

def getStd(dimension):
    mean = np.mean(dimension)
    sigma = 0
    for elem in dimension:
        sigma += abs((elem - mean)**2)
    
    std = math.sqrt(sigma/len(dimension))

    return std
