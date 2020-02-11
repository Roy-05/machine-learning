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
    pt_class = 1 if y - m*x - c >= 0 else -1
    dataset.append([x, y, pt_class])

def activate(s):
    return 1 if s >= 0 else -1

def feedForward(row):
    sigma = weights[0]
    for weight,x in zip(weights[1:], row):
        sigma += weight*x
    
    return activate(sigma)

for row in dataset:
    e = feedForward(row)
    print(f"Expect: {row[-1]} Actual: {e}")



