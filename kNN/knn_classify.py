from sys import argv
import numpy as np

script, pendigits_training, pendigits_test = argv

training_dataset = open(f"{pendigits_training}.txt", 'r')

matrix = []

for line in training_dataset:
    matrix.append(line.split())

print(matrix[0])