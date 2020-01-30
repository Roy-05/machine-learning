from random import randint, random

x_points = []
y_points = []
w = randint(1, 10)
b = randint(1, 10)
datapoints = 20
epochs = 50
learning_rate = 0.000001

for i in range(datapoints):

    x_points.append(randint(1,50))


for x in x_points:

    # y = w * x + b
    # Yn = y +/- randint(0.1 * y)
    y = w * x + b
    y_points.append(y + (-1)**randint(0,1) * randint(0,round(0.1 * y)))

w1 = random()
b1 = random()

def update_weight_and_bias(w, b):
    w_derive = 0
    b_derive = 0
    for i in range(datapoints):
        w_derive += -2 * x_points[i] * (y_points[i] - (w * x_points[i] + b))
        b_derive += -2 * (y_points[i] - (w * x_points[i] + b))
    
    w -= (w_derive / datapoints) * learning_rate
    b -= (b_derive / datapoints) * learning_rate

    return w,b

for i in range(epochs):
    w1,b1 = update_weight_and_bias(w1, b1)
    print(w1, b1)
    
