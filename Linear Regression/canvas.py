import regressor as rg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random

fig = plt.figure(num = "Linear Regression with Gradient Descent Algorithm")

X0 = min(rg.x_points) 
Y0 = X0 * rg.w + rg.b

Xn = max(rg.x_points)
Yn = Xn * rg.w + rg.b

w1 = random()
b1 = random()

Ymin = min(Y0, min(rg.y_points), (Xn * w1 + b1))
Ymax = max(Yn, max(rg.y_points))

ax = plt.axes(xlim = (X0 - 5, Xn + 5), ylim = (Ymin - 5 , Ymax + 5))

ax.plot([X0, Xn], [Y0, Yn], color = "green", lw = 3, zorder= 1)
ax.plot(rg.x_points, rg.y_points, 'ro', zorder = 2)

[line] = ax.plot([],[],lw=3, color = "#f9a602", zorder = 3)

points = []

for i in range(rg.epochs):
    dW, dB, mse = rg.get_accumulated_errors_and_mse(w1, b1)
    w1 += (2/rg.datapoints) * dW * rg.learning_rate
    b1 += (2/rg.datapoints) * dB * rg.learning_rate

    print("Epoch:{0:4d} and Mean Squared Error: {1:6.8f}".format(i+1,mse))
    Yn = Xn * w1 + b1
    points.append([[X0, Xn], [Y0, Yn]])

def animate(i):
    if(i>=len(points)):
        plt.pause(1)
        plt.close('all')
    else:
        line.set_data(points[i][0], points[i][1])

anim = FuncAnimation(
        fig, 
        animate, 
        frames = rg.epochs+1, 
        interval = 100)

plt.show()
