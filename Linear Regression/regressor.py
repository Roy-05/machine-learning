from random import randint, random

x_points = []
y_points = []
w = randint(1, 5)
b = randint(1, 10)
datapoints = 20
epochs = 100
learning_rate = 0.00001

for i in range(datapoints):
    x_points.append(randint(1,50))
    
    y = w * x_points[i] + b
    y_points.append(y + (-1)**randint(0,1) * randint(0,round(0.1 * y)))


def get_accumulated_errors_and_mse(w, b):

    dW = 0
    dB = 0
    mse = 0
    for i in range(datapoints):
        y = (w * x_points[i] + b)
        dW += x_points[i] * (y_points[i] - y)
        dB += y_points[i] - y
        mse += ((y_points[i] - y)**2)/datapoints

    return dW, dB, mse
