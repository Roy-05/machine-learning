from random import random, randint, uniform
import math

#Slope ranges from  PI/6 - PI/3
m = uniform(math.pi/6, math.pi/3)
c = randint(-50,50)

learning_rate = 0.0001
datapoints = 20
epochs = 0
points = []
dataset = []

#Initialize weights with bias [i.e. w1 = bias]
weights = [random(), random(), random()] 

#Generatate dataset
for i in range(datapoints):
    x = randint(50,950)
    y = randint(50,950)
    pt_class = 1.0 if y - m*x - c >= 0.00 else -1.0
    dataset.append([1, x, y, pt_class])


#Activation function is a threshold function.
def activate(s):
    return 1.0 if s >= 0.0 else -1.0

def feedForward(row):
    sigma = 0
    for weight,x in zip(weights, row):
        sigma += weight*x
    
    return activate(sigma)

def train(row):
    output = feedForward(row[0:3])
    for i in range(len(row)-1):
        weights[i] += learning_rate * (row[-1] - output) * row[i]

def getCurrentPoints(points):
    """
    Derived from w1x1 + w2x2 + w3x3 = 0
    where w1 = bias and x1 = 1
    plugging x2 = 0, x3 = - w1x1/w3
    and x2 = 1000, x3 = - (w2x2 + w1x1)/w3
    """

    x0 = 0
    y0 = - weights[0]/weights[2]
    xn = 1000
    yn = - (weights[1] * 1000 + weights[0])/ weights[2]

    points.append([[x0,xn], [y0,yn]])

while(True):
    epochs += 1
    converged = True
    misclassify = 0

    for row in dataset:
        classify = feedForward(row[0:3])
        train(row)

        if(row[-1] != classify):
            converged = False
            misclassify += 1

    getCurrentPoints(points) 
    print(f"Epoch:{epochs:6} misclassified:{misclassify:2}\n")
    if(converged):
        break 
   