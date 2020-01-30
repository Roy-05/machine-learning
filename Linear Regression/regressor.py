from random import randint, random


x_points = []
y_points = []
w = randint(1, 5)
b = randint(1, 10)
datapoints = 20
epochs = 1000
learning_rate = 0.000001

for i in range(datapoints):
    x_points.append(randint(1,50))
    
    y = w * x_points[i] + b
    y_points.append(y + (-1)**randint(0,1) * randint(0,round(0.1 * y)))


def get_accumulated_errors(w, b):

    dW = 0
    dB = 0
    for i in range(datapoints):
        dW += x_points[i] * (y_points[i] - (w * x_points[i] + b))
        dB += y_points[i] - (w * x_points[i] + b)
    
    return dW, dB
