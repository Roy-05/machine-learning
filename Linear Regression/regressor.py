from random import randint, random


x_points = []
y_points = []
w = randint(1, 5)
b = randint(1, 10)
datapoints = 20
epochs = 50
learning_rate = 0.0001

for i in range(datapoints):
    x_points.append(randint(1,50))
    
    y = w * x_points[i] + b
    y_points.append(y + (-1)**randint(0,1) * randint(0,round(0.1 * y)))


w1 = random()
b1 = random()

def update_weight_and_bias(w, b):
    w_derive = 0
    b_derive = 0
    for i in range(datapoints):
        w_derive += x_points[i] * (y_points[i] - (w * x_points[i] + b))
        b_derive += y_points[i] - (w * x_points[i] + b)
    
    w -= (-2 * w_derive / float(datapoints)) * learning_rate
    b -= (-2 * b_derive / float(datapoints)) * learning_rate

    return w,b
