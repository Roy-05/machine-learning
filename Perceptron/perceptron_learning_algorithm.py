from random import random, randint, uniform
import math

#Slope ranges from  PI/12 - 5PI/12
m = uniform(math.pi/12, 5*math.pi/12)
c = randint(-50,50)
learning_rate = 0.0001

dataset = []
weights = [random(), random(), random()] #Initialize with Bias

print(weights)
#Generatate Datapoints
for i in range(20):
    x = randint(50,950)
    y = randint(50,950)
    pt_class = 1.00 if y - m*x - c >= 0.00 else -1.00
    dataset.append([1, x, y, pt_class])

def activate(s):
    return 1.00 if s >= 0.00 else -1.00

def feedForward(row):
    sigma = weights[0]
    for weight,x in zip(weights, row):
        sigma += weight*x
    
    return activate(sigma)

def train(row):
    output = feedForward(row)
    for i in range(len(weights)):
        weights[i] += learning_rate * (row[-1] - output) * row[i]

